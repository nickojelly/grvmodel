import os
import pickle

import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
import torch
import torch.optim as optim
import wandb
from rnn_tools.rnn_classes import *
import time
import numpy as np
from rnn_tools.model_saver import model_saver_wandb
from torchviz import make_dot
from rnn_tools.model_saver import model_saver_wandb


def validation_CLE(x,y):
    loss_t = -torch.log(torch.exp(x)/torch.sum(torch.exp(x), dim=-1, keepdim=True))*y
    return loss_t

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)

def batch_indexer(batch_size,group_len, dataset_len):
    num_batches = group_len
    group_size = batch_size*group_len
    all_batches = []
    for n in range(num_batches):
        batch = []
        
        for i in range(dataset_len):
            if batch_size*(n)<i%group_size<batch_size*(n+1):
                batch.append(i)
        
        all_batches.append(batch)
    return all_batches

def save_wandb_tables(df:pd.DataFrame, table_name):
    model_name = wandb.run.name
    df.to_csv(f'./wandbTables/{model_name}/{model_name}_{table_name}.csv')
    isExist = os.path.exists(
        f"./wandbTables/{model_name}/"
    )


def train_quick_pass(model:GRUNetv3,raceDB,config,save_output=False):
    num_batches = raceDB.batches['num_batches']
    hidden_state_init = model.h0
    raceDB.reset_hidden_w_param(hidden_state_init,num_layers=2, hidden_size=config['hidden_size'])
        
    raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(config['hidden_size']+config['f1_layer_size'])).to('cuda:0')
    raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(config['hidden_size'])).to('cuda:0')
    # raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(config['hidden_size']+config['input_size'])).to('cuda:0')
    for i in range(num_batches):
        dogs = raceDB.batches['dogs'][i]
        batch_races = raceDB.batches['batch_races'][i]
        train_dog_input = raceDB.batches['train_dog_input'][i]
        track_hashes = torch.tensor([x.track_hash for x in batch_races]).to('cuda:0')
        X = raceDB.batches['packed_x'][i]
        X_d = raceDB.packed_x_data[i]
        hidden_in = torch.stack([x.hidden for x in dogs]).transpose(0,1)
        output,hidden = model((X,X_d), h=hidden_in)
        hidden = hidden.transpose(0,1)
        [setattr(obj, 'hidden', val) for obj, val in zip(dogs,hidden)]

        if save_output:
            for j,dog in enumerate(train_dog_input):
                [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[j])]
            X2 = torch.stack([torch.cat([race.race_dist]+[d.hidden_out for d in race.dogs])  for race in batch_races])

            output,_,output_p = model(X2, p1=False,hashes=track_hashes)

            for j,race_e in enumerate(batch_races):
                race_e.output = (output[j].detach(),output_p[j].detach())
        
    


def quick_profitability(prices,classes,output):

    sft_max = nn.Softmax(dim=-1)
    softmax_preds = sft_max(output)
    _, actual = torch.max(classes.data, 1)
    onehot_win = F.one_hot(actual, num_classes=8)
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == actual).sum().item()
    label = torch.zeros_like(classes.data).scatter_(1, torch.argmax(classes.data, dim=1).unsqueeze(1), 1.).requires_grad_(True)
    pred_label = torch.zeros_like(output.data).scatter_(1, torch.argmax(output.data, dim=1).unsqueeze(1), 1.).requires_grad_(True)
    correct_tensor = label*pred_label
    
    price_tensor = torch.tensor(prices, device='cuda:0',requires_grad=True)
    delta_tensor = softmax_preds-(1/price_tensor)
    delta_tensor_clip = torch.clamp(delta_tensor,0,1)
    
    win_price = price_tensor*label
    win_price = win_price.max(dim=1)[0]
    profit_tensor = price_tensor*label*delta_tensor_clip-delta_tensor_clip

    # print(f"{correct=}\n{profit_tensor=}\n{softmax_preds=}\n{delta_tensor_clip=}\n{1/price_tensor=}")
    # Create a mask of the NaN values in profit_tensor
    mask = torch.isnan(profit_tensor)

    # Use the mask to filter out the NaN values
    profit_tensor = profit_tensor[~mask]

    # print(f"{correct=}\n{profit_tensor=}\n{win_price=}")

    return correct, profit_tensor*-1, win_price

def simple_profit(simple_model,prices,classes,output,output_p, shuffle = False, shuffle_idx = None):
    prices = prices
    # Shuffle the data of the last dimension
    if shuffle:
        idx = shuffle_idx
        # output = output[:,idx]
        output_p = output_p[:,idx]
        classes = classes[:,idx]
        prices = prices[:,idx]
    # print(f"{output=}\n{output_p=}\n{classes=}")
    # print(f"{output.shape=}")
    bet_amounts = simple_model(output,output_p,prices.nan_to_num(0,0,0).detach())

    price_over_30_mask = prices<30
    price_over_30_mask = price_over_30_mask.to(float).nan_to_num(0,0,0)
    classes = classes*price_over_30_mask

    label = torch.zeros_like(classes.data).scatter_(1, torch.argmax(classes.data, dim=1).unsqueeze(1), 1.).requires_grad_(True)
    profit_tensor = prices.nan_to_num(0,0,0)*label*bet_amounts.nan_to_num(0,0,0)-bet_amounts.nan_to_num(0,0,0)
    # print(f"{profit_tensor=}")

    return profit_tensor
def betfair_log_loss(output, classes, probabilites,reduce=True):
    label = torch.zeros_like(classes.data).scatter_(1, torch.argmax(classes.data, dim=1).unsqueeze(1), 1.)
    output_prob = F.softmax(output,dim=-1)
    if reduce:
        log_loss = (-(label*output_prob.log()+(1-label)*(1-output_prob).log())).sum(dim=-1)
        betfair_loss = (-(label*probabilites.log()+(1-label)*(1-probabilites).log())).sum(dim=-1)
        nan_mask = torch.isnan(betfair_loss)
        betfair_loss[nan_mask] = 0
        log_loss[nan_mask] = 0
        return log_loss.mean(), betfair_loss.mean()
    else:
        log_loss = (-(label*output_prob.log()+(1-label)*(1-output_prob).log()))
        betfair_loss = (-(label*probabilites.log()+(1-label)*(1-probabilites).log()))
        nan_mask = torch.isnan(betfair_loss)
        betfair_loss[nan_mask] = 1.0
        log_loss[nan_mask] = torch.tensor(1.0,requires_grad=True,device='cuda:0')  
        return log_loss, betfair_loss, label

def train_double_v3(model:GRUNetv3_extra_embedding,raceDB:Races, criterion, optimizer,scheduler, config=None,update=False,save_output=False):
    # torch.autograd.set_detect_anomaly(True)

    epochs = config['epochs']
    test_max_roi,val_max_roi,val_loss_min = -1,-1,100

    num_batches = raceDB.batches['num_batches']
    example_ct = 0  # number of examples seen
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    raceDB.hidden_state_inits = []

    profit_model = GRUNetv3_profit(raceDB).to('cuda:0')
    profit_optim = optim.Adam(profit_model.parameters(), lr=0.001,maximize=True)

    ohe_criterion = nn.CrossEntropyLoss()

    raceDB.profit_model = profit_model
    ratio = config['price_ratio']


    for epoch in trange(epochs):
        model.train()
        if update:
            model.eval()
        hidden_state_init = model.h0.requires_grad_(True)
        raceDB.reset_hidden_w_param(hidden_state_init,num_layers=2, hidden_size=config['hidden_size'])
        raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(config['hidden_size']+config['f1_layer_size'])).to('cuda:0')
        raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(config['hidden_size'])).to('cuda:0')
        # raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(config['hidden_size']+config['input_size'])).to('cuda:0')
        for i in range(num_batches):
            with torch.cuda.amp.autocast():
                dogs = raceDB.batches['dogs'][i]
                train_dog_input = raceDB.batches['train_dog_input'][i]
                # train_dog_input_np = raceDB.batches['train_dog_input_np'][i]
                batch_races = raceDB.batches['batch_races'][i]
                batch_races_ids = raceDB.batches['batch_races_ids'][i]
                X = raceDB.batches['packed_x'][i]
                track_hashes = torch.tensor([x.track_hash for x in batch_races]).to('cuda:0')
                X_d = raceDB.packed_x_data[i]

                example_ct+=len(batch_races)
                hidden_in = torch.stack([x.hidden for x in dogs]).transpose(0,1)
                output,hidden = model((X,X_d), h=hidden_in)
                
                hidden = hidden.transpose(0,1)


                for j,dog in enumerate(train_dog_input):
                    [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[j])]

                [setattr(obj, 'hidden', val) for obj, val in zip(dogs,hidden)]

                # [setattr(race, 'hidden_in', torch.cat([race.race_dist]+[d.hidden_out for d in race.dogs])) for race in batch_races]

                X2 = torch.stack([torch.cat([race.race_dist]+[d.hidden_out for d in race.dogs])  for race in batch_races])
                
                race = batch_races

                # X2 = torch.stack([r.hidden_in for r in race]) #Input for FFNN
                y = torch.stack([x.classes for x in race])
                y_ohe = torch.stack([x.one_hot_class for x in race])
                
                y_p = torch.stack([x.start_prob for x in race])
                lw = torch.stack([x.loss.detach() for x in race]).requires_grad_(True).detach()
                lw_log = torch.stack([x.loss_log.detach() for x in race]).requires_grad_(True).detach()
                w = torch.stack([x.new_win_weight for x in race])
                p = torch.stack([torch.tensor(x.prices,device='cuda:0') for x in race])

                idx = torch.randperm(8)+1
                # X2_split = X2.hsplit([1+x*(config['hidden_size']+config['f1_layer_size']) for x in range(0,8)])
                X2_split = X2.hsplit([1+x*(config['hidden_size']) for x in range(0,8)])
                X2_split_arrange = [X2_split[x] for x in idx]
                X2 = torch.cat([X2_split[0]]+X2_split_arrange,dim=-1) 
                
                output,_,output_p = model(X2, p1=False,hashes=track_hashes)

                y = y[:,idx-1]
                y_p = y_p[:,idx-1]
                y_ohe = y_ohe[:,idx-1]

                y_ohe_arg = y_ohe.argmax(dim=-1)

                y_mean = (y*(1-ratio)+y_p*ratio)

                
                betfair_probs = torch.stack([x.implied_prob for x in race],dim=0)
                betfair_probs = betfair_probs[:,idx-1]
                log_loss, betfair_loss = betfair_log_loss(output, y, betfair_probs)

                log_loss_weighted = log_loss - betfair_loss
                log_loss_mask = log_loss_weighted>0
                log_loss_masked = log_loss[log_loss_mask]
                

                log_loss_unreduced, _,_  = betfair_log_loss(output, y, betfair_probs, reduce=False)

                # log_loss = (log_loss_unreduced.sum(-1)*lw_log).mean()
                # log_loss = log_loss.mean()

                # print(log_loss_unreduced.sum(-1).requires_grad_(True))

                epoch_loss = criterion(output, y_mean)*lw
                epoch_loss_ohe = criterion(output, y_ohe_arg)*lw
                epoch_loss_p = criterion(output_p, y_p)*lw


                [setattr(race, 'loss', loss.mean().detach()) for race,loss in zip(batch_races,criterion(output, y_p))]
                [setattr(race, 'loss_log', loss.mean().detach()) for race,loss in zip(batch_races,log_loss_unreduced.sum(-1).requires_grad_(True))]
                # raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(config['hidden_size']+config['input_size'])).to('cuda:0')

            t6 = time.perf_counter()
            if not update and epoch > 0:
                optimizer.zero_grad()
                model.zero_grad()
                # profit_optim.zero_grad()
                # log_loss.mean().backward()
                # ((epoch_loss+5*epoch_loss_p)).mean().backward()
                # epoch_loss_ohe.mean().backward()
                epoch_loss_p.mean().backward()  
                # epoch_loss.mean().backward()
                
                optimizer.step()
                wandb.log({"loss_1": torch.mean(epoch_loss).item(), 'epoch':epoch, 'ohe_loss':torch.mean(epoch_loss_ohe).item()})
            raceDB.detach_hidden(dogs)
            t7 = time.perf_counter()
        if epoch%5 == 0:
            t0 = time.perf_counter()
            quick_pass_val(model,raceDB, criterion=criterion, epoch=epoch)
            t1 = time.perf_counter()
            print(f"Quickpass time {t1-t0}")
            torch.cuda.empty_cache()
        if (epoch)%20==0:
            t8 = time.perf_counter()
            model = model.eval()
            train_quick_pass(model,raceDB,config)
            t0 = time.perf_counter()
            test_stats = test_model_v3(model,raceDB, criterion=criterion, epoch=epoch)
            t1 = time.perf_counter()
            val_stats = validate_model_v3(model,raceDB, criterion=criterion, epoch=epoch)
            t2 = time.perf_counter()
            print(f"Test time : {t1-t0}, Val time : {t2-t1}, Total time : {t2-t0}")
            if (test_stats['ROI < 30'].round(4) > test_max_roi or val_stats['val_ROI < 30'].round(4)  > val_max_roi) or round(val_stats['val_loss_val'],5)  < val_loss_min:
                
                test_max_roi = round(max(test_stats['ROI < 30'],test_max_roi),4)
                val_max_roi =  round(max(val_stats['val_ROI < 30'],val_max_roi),4)
                val_loss_min = round(min(val_stats['val_loss_val'],val_loss_min),5)
                print(f"New Max ROI: {round(test_stats['ROI < 30'],4)}, {round(val_stats['val_ROI < 30'],4)}, {round(val_stats['val_loss_val'],5)}")
                raceDB.create_hidden_states_dict_v2()
                model_saver_wandb(model, optimizer, epoch, test_max_roi, raceDB.hidden_states_dict_gru_v6, raceDB.train_hidden_dict, model_name="long nsw new  22000 RUN")
            t9 = time.perf_counter()
        if not update:
            hidden_state_init = model.h0
            raceDB.reset_hidden_w_param(hidden_state_init,num_layers=2, hidden_size=config['hidden_size'])
        else: 
            return model
        torch.cuda.empty_cache()

    return model

def kelly_criterion(odds, probabilities,D=0.95):
    """
    Calculate the Kelly Criterion bet sizes for multiple outcomes.

    Parameters:
    D (float): The dividend rate.
    odds (list): The odds for each outcome.
    probabilities (list): The estimated probability of each outcome.

    Returns:
    list: The fraction of the bankroll to bet on each outcome.
    """

    # Convert odds to decimal and calculate betas
    odds = [1 / o for o in odds]
    betas = [1 / o for o in odds]

    # Calculate the expected revenue rate for each outcome
    er = [(D * p) / b for p, b in zip(probabilities, betas)]

    # Initialize the set of outcomes to bet on and its revenue rate
    S = set()
    R_S = 1

    # Sort the outcomes by expected revenue rate in descending order
    outcomes = sorted(range(len(er)), key=lambda i: -er[i])

    for k in outcomes:
        # If the expected revenue rate of the k-th outcome is greater than the current revenue rate of the set...
        if er[k] > R_S:
            # ...then add the k-th outcome to the set and recalculate its revenue rate
            S.add(k)
            R_S = D * sum(probabilities[i] for i in range(len(probabilities)) if i not in S) / (D - sum(betas[i] for i in S))

    # Calculate the optimal fraction to bet on each outcome in the set
    f = [p - b * sum(probabilities[i] for i in range(len(probabilities)) if i not in S) / (D - sum(betas[i] for i in S)) if k in S else 0 for k, p, b in zip(range(len(probabilities)), probabilities, betas)]

    return f

def apply_kelly_to_df(df):
    # Group the DataFrame by raceid
    grouped = df.groupby('raceID')

    # Initialize a new column in the DataFrame for the Kelly ratios
    df['seq_kelly_ratio'] = 0.0

    # Iterate over each group
    for name, group in grouped:
        # Extract the odds and probabilities for this group
        odds = group['imp_prob'].tolist()
        probabilities = group['pred_prob'].tolist()

        # Calculate the dividend rate
        D = 1 - 0.1

        # Apply the Kelly Criterion to this group
        kelly_ratios = kelly_criterion(odds, probabilities, D=D)

        # Update the 'seq_kelly_ratio' column in the DataFrame for this group
        df.loc[group.index, 'seq_kelly_ratio'] = kelly_ratios

    return df


def clean_data(df):
    kl_div = nn.KLDivLoss(reduction='batchmean')

    df['imp_prob'] = 1 / df['prices']
    df['pred_price'] = np.clip(df['pred_price'], 0, 100)
    df['pred_prob'] = 1 / df['pred_price']
    df['pred_prob2'] = 1 / df['pred_price2']

    df['bet_amount'] = np.where(
        (df['pred_prob'] > df['imp_prob']) & (df['imp_prob'] > 0) & (df['imp_prob'] < 1),
        (df['pred_prob'] - df['imp_prob']) * 100,
        0
    )
    df['profit'] = np.where(
        df['onehot_win'],
        df['bet_amount'] * (df['prices'] - 1) * 0.95,
        -1 * df['bet_amount']
    )
    # df['win_price'] = np.where(df['onehot_win'], df['prices'], 0)
    df = apply_kelly_to_df(df)
    df['bet_kelly_seq'] = np.where(
                            df['prices']<30,
                            df['seq_kelly_ratio']*100,
                            0
                        )
    
    df['profit_kelly_sq'] = np.where(
                    df['onehot_win'],
                    df['bet_kelly_seq']  * (df['prices'] - 1) * 0.95,
                    -1 * df['bet_kelly_seq']
                )

    df['colour'] = np.where(
        df['profit'] > 0,
        "profitz",
        np.where(
            df['profit'] < 0,
            "loss",
            np.where(df['onehot_win'], "no bet - win", "no bet")
        )
    )

    df['profit < 30'] = np.where(df['prices'] < 30, df['profit'], 0)
    df['outlay < 30'] = np.where(df['prices'] < 30, df['bet_amount'], 0)

    df['bet_amount2'] = np.where(
        (df['pred_prob2'] > df['imp_prob']) & (df['imp_prob'] > 0) & (df['imp_prob'] < 1),
        (df['pred_prob2'] - df['imp_prob']) * 100,
        0
    )
    df['profit2'] = np.where(
        df['onehot_win'],
        df['bet_amount2'] * (df['prices'] - 1) * 0.95,
        -1 * df['bet_amount2']
    )
    df['bet_amount_kelly'] = np.where(
        (df['pred_prob'] > df['imp_prob']) & (df['imp_prob'] > 0) & (df['imp_prob'] < 1),
        ((df['pred_prob']*(df['prices']*0.95-1)-(1-df['pred_prob']))/(df['prices']*0.95-1)) * 100,
        0
    )
    df['profit_kelly'] = np.where(
        df['onehot_win'],
        df['bet_amount_kelly']  * (df['prices'] - 1) * 0.95,
        -1 * df['bet_amount_kelly'] 
    )
    df['profit < 30 2'] = np.where(df['prices'] < 30, df['profit2'], 0)
    df['outlay < 30 2'] = np.where(df['prices'] < 30, df['bet_amount2'], 0)

    df['bet_relu'] = df['bet_amount'] * 1/df['mutual_info']
    df['profit_relu'] = np.where(
        df['onehot_win'],
        df['bet_relu'] * (df['prices'] - 1) * 0.95,
        -1 * df['bet_relu']
    )

    df['bet_relu<30'] = np.where(df['prices'] < 20, df['bet_amount'] * df['relu'], 0)
    df['profit_relu<30'] = np.where(df['prices'] < 20, df['profit_relu'], 0)
    df['kl_div'] = kl_div(torch.tensor(df['pred_prob'].tolist()),torch.tensor(df['classes'].tolist())).tolist()
    df['kl_div_bfsp'] = kl_div(torch.tensor(df['imp_prob'].tolist()),torch.tensor(df['classes'].tolist())).tolist()
    df['win < 30'] = np.where(df['profit < 30']>0 ,1,0)
    df['win'] = np.where(df['profit']>0 ,1,0)
    df['bet_count'] = np.where(df['bet_amount']>0 ,1,0)
    df['bet_count < 30'] = np.where(df['outlay < 30'] > 0 ,1,0)   

    df['profit_model'] = np.where(
        df['onehot_win'],
        df['bet_amount_model']  * (df['prices'] - 1) * 0.95,
        -1 * df['bet_amount_model']
    )

    

    return df

import sys

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            # print(m.__class__.__name__)
            m.train()

def get_monte_carlo_predictions(data,
                                hashes,
                                forward_passes,
                                model,
                                n_classes,
                                n_samples):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        with torch.no_grad():
            output,relu,output_p,= model(data,hashes=hashes, p1=False)
            if i == 0:
                # print(f"{output.shape=}")
                pass
            output = softmax(output)  # shape (n_samples, n_classes)
        predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes 
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                           axis=-1), axis=0)  # shape (n_samples,)
    
    # print(f"{mean.shape=}\n{variance.shape=}\n{entropy.shape=}\n{mutual_info.shape=}")

    return mean, variance, entropy, mutual_info

def quick_pass_val(model:GRUNetv3_extra_embedding,raceDB:Races,criterion,epoch,device='cuda:0'):
    #Testing
    model.eval()
    with torch.no_grad():
        len_test = len(raceDB.test_dog_ids)
        test_idx = range(0,len_test)
        dogs = [x for x in  raceDB.test_dogs.values()]  #[Dog]
        dog_input = [inner for inner in [x for x in raceDB.get_dog_test(test_idx)]] #[[DogInput]]
        X = raceDB.batches['packed_y']
        X_d = raceDB.packed_y_data
        hidden_in = torch.stack([x.hidden for x in dogs]).transpose(0,1)
        output,hidden = model((X,X_d),h=hidden_in)

        hidden = hidden.transpose(0,1)
        for i,dog in enumerate(dogs):
            dog.hidden_test = hidden[i]


        for i,dog in enumerate(dog_input):
                [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[i])]

        # raceDB.margin_from_dog_to_race_v3(mode='test')
        len_test = len(raceDB.test_race_ids)
        test_idx = range(0,len_test)
        race = raceDB.get_test_input(test_idx)

        Xt = torch.stack([torch.cat([race.race_dist]+[d.hidden_out for d in race.dogs])  for race in race])
        y = torch.stack([x.classes for x in race])
        track_hashes = torch.tensor([x.track_hash for x in race]).to('cuda:0')

        output,relu,output_p,= model(Xt, p1=False,hashes=track_hashes)

        loss_test = criterion(output, y).mean()

        betfair_probs = torch.stack([x.implied_prob for x in race],dim=0)
        # print(f"")
        log_loss_test, betfair_loss_test = betfair_log_loss(output, y, betfair_probs)

        
    #Validation
    with torch.no_grad():
        len_test = len(raceDB.val_dog_ids)
        val_idx = range(0,len_test)
        dogs = [x for x in  raceDB.val_dogs.values()]  #[Dog]
        dog_input = [inner for inner in [x for x in raceDB.get_dog_val(val_idx)]] #[[DogInput]]
    
        X = raceDB.batches['packed_v']
        X_d = raceDB.packed_v_data

        hidden_in = torch.stack([x.hidden_test for x in dogs]).transpose(0,1)


        output,hidden = model((X,X_d),h=hidden_in) # Shape List[Tensor[Dog]]
        hidden = hidden.transpose(0,1)
        for i,dog in enumerate(dogs):
            dog.hidden_test = hidden[i]


        for i,dog in enumerate(dog_input):
                [setattr(obj, 'hidden_out', val.detach()) for obj, val in zip(dog,output[i])]


        len_test = len(raceDB.val_race_ids)
        val_idx_races = range(0,len(raceDB.val_race_ids))
        race = raceDB.get_val_input(val_idx_races)

        Xt = torch.stack([torch.cat([race.race_dist]+[d.hidden_out for d in race.dogs])  for race in race])
        y = torch.stack([x.classes for x in race])
        track_hashes = torch.tensor([x.track_hash for x in race]).to('cuda:0')

        output,relu,output_p,= model(Xt, p1=False,hashes=track_hashes)

        loss_val = criterion(output, y).mean()

        betfair_probs = torch.stack([x.implied_prob for x in race],dim=0)
        # print(f"")
        log_loss_val, betfair_loss_val = betfair_log_loss(output, y, betfair_probs)

        wandb.log({"q_loss_test": loss_test.item(),
                    "q_loss_val": loss_val.item(),
                    "q_betfair_loss_test": betfair_loss_test.item(), 
                    "q_betfair_loss_val": betfair_loss_val.item(),
                    "q_log_loss_val": log_loss_val.item(),
                    "q_log_loss_test": log_loss_test.item(),
                    "epoch":epoch,})

def validate_model_pass(model:GRUNetv3_extra_embedding,raceDB:Races,race, criterion,test_idx,device='cuda:0',save_output=False):

    profit_model = raceDB.profit_model
    sft_max = nn.Softmax(dim=-1)
    # distances = 
    Xt = torch.stack([torch.cat([race.race_dist]+[d.hidden_out for d in race.dogs])  for race in race]) #Input for FFNN
    y = torch.stack([x.classes for x in race])
    y_p = torch.stack([x.prob for x in race])
    y_ohe = torch.stack([x.one_hot_class for x in race])
    y_bfsp = torch.stack([x.implied_prob for x in race])    
    r = torch.stack([r.implied_prob for r in race])
    p = torch.stack([torch.tensor(x.prices,device='cuda:0') for x in race])

    track_hashes = torch.tensor([x.track_hash for x in race]).to('cuda:0')
    # idx = torch.randperm(len(track_hashes))
    # track_hashes = track_hashes[idx]

    output,relu,output_p,= model(Xt, p1=False,hashes=track_hashes)
    mean, variance, entropy, mutual_info = get_monte_carlo_predictions(Xt,track_hashes, 100,model,8,y.shape[0])
    # print(f"{mean=}\n{variance=}\n{entropy=}\n{mutual_info=}")
    if save_output:
        for j,race_e in enumerate(race):
            race_e.output = (output[j].detach(),output_p[j].detach())

    kl = nn.KLDivLoss(reduction='none')
    
    relu = r.mean(dim=1)

    _, actual = torch.max(y.data, 1)
    onehot_win = F.one_hot(actual, num_classes=8)
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == actual).sum().item()

    _, favorite = torch.max(y_bfsp, 1)
    correct_bfsp = (favorite == actual).sum().item()

    #One hot wins
    label = torch.zeros_like(y.data).scatter_(1, torch.argmax(y.data, dim=1).unsqueeze(1), 1.)
    pred_label = torch.zeros_like(output.data).scatter_(1, torch.argmax(output.data, dim=1).unsqueeze(1), 1.)
    favorite_label = torch.zeros_like(y_bfsp).scatter_(1, torch.argmax(y_bfsp, dim=1).unsqueeze(1), 1.)
    correct_tensor = label*pred_label
    favorite_tensor = label*favorite_label

    correct_l = predicted == actual

    softmax_preds = sft_max(output)
    # softmax_preds1 = sft_max(output/0.5)
    softmax_preds2 = sft_max(output/0.75)

    softmin_preds = F.softmin(output,dim=-1)

    loss = criterion(output, y).mean()
    loss_ohe = criterion(output, y_ohe.float()).mean()
    loss_p = criterion(output_p, y_p).mean()
    loss_bfsp = criterion(y_bfsp, y).mean()

    loss_kl = kl(softmax_preds.log(), y)
    loss_kl_bfsp = kl(y_bfsp.log(), y)

    loss_tensor = validation_CLE(output,y)
    loss_tensor_bfsp = validation_CLE(y_bfsp,y)

    test_races = race

    price_tensor = torch.tensor([x.prices for x in test_races], device=device)
    win_price = price_tensor*label
    win_price = win_price.max(dim=1)[0]
    profit_tensor = price_tensor*correct_tensor*0.95-pred_label

    bet_amount = profit_model(output,output_p,p)

    bet_amount = bet_amount.nan_to_num(0)

    betfair_probs = torch.stack([x.implied_prob for x in race],dim=0).to('cuda:0')
    log_loss, betfair_loss = betfair_log_loss(output, y, betfair_probs)
    bet_order = softmin_preds.argsort(dim=-1).argsort(dim=-1)

    log_loss_unreduced, betfair_loss_unreduced, label_loss = betfair_log_loss(output, y, betfair_probs, reduce=False)
    races = {}
    races['raw_margins'] = [x.raw_margins for x in test_races]
    races['correct'] = correct_tensor.tolist()
    races['simple']  = profit_tensor.tolist()
    # races['output'] = [x.margins for x in test_races]
    races['win_price'] = [[x.win_price_weight.cpu().item()]*8 for x in test_races]   
    # print(races['win_price'])
    # dasd
    races['relu'] = [[x]*8 for x in relu.cpu().tolist()]
    # print(races['relu'])
    races['pred_order'] = bet_order.tolist()
    races['output_logits'] = output.tolist()
    races['bet_amount_model'] = bet_amount.tolist()
    races['output_price'] = (1/sft_max(output_p)).tolist()
    races['pred_prob'] = softmax_preds.tolist()
    races['pred_prob2'] = softmax_preds2.tolist()
    races['prices'] = [x.prices for x in test_races]
    races['imp_prob'] = [x.implied_prob.tolist()  for x in test_races]
    races['betfair_log_loss']  = betfair_loss_unreduced.tolist()
    races['log_loss'] = log_loss_unreduced.tolist()
    races['label_loss'] = label_loss.tolist()
    races['pred_price'] = (1/softmax_preds).tolist()
    # races['pred_price1'] = (1/softmax_preds1).tolist()
    
    races['pred_price2'] = (1/softmax_preds2).tolist()
    # races['pred_prob'] = [x.tolist() for x in races['pred_prob']]
    races['classes'] = [x.classes.tolist() for x in test_races]
    races['track'] = [x.track_name for x in test_races]
    races['onehot_win'] = [x.one_hot_class.tolist() for x in test_races]
    # races['date'] = [x.race_date for x in test_races]
    races['dogID'] = [x.list_dog_ids() for x in race]
    races['dog_name'] = [x.list_dog_names() for x in race]
    races['dog_box'] = [[0,1,2,3,4,5,6,7] for x in race]
    races['raceID'] = [[x.raceid]*8 for x in race]
    races['date'] = [[x.race_date]*8 for x in race]
    races['entropy'] = [[x]*8 for x in entropy.tolist()]
    races['mutual_info'] = [[x]*8 for x in mutual_info.tolist()]
    races['race_num'] = [[int(x.race_num)]*8 for x in race]
    races['loss'] = loss_kl.tolist()
    races['loss_bfsp'] = loss_kl_bfsp.tolist()
    races['correct'] = correct_tensor.tolist()
    races['favorite_correct'] = favorite_tensor.tolist()


    accuracy = correct/len(predicted)

    # for k,v in races.items():
    #     print(f"{k} length is {len(v)} type {type(v[0])}")

    

    races['one_hot_win'] = onehot_win.tolist()
    races['track'] = [[x.track_name]*8 for x in test_races]

    for k,value in races.items():
        races[k] = [item for sublist in value for item in sublist]
    all_price_df = pd.DataFrame(races)


    return all_price_df, loss, loss_p,loss_kl_bfsp,loss_kl, correct, accuracy,mutual_info,log_loss, betfair_loss, loss_ohe

#Testing
@torch.no_grad()
def test_model_v3(model:GRUNetv3,raceDB:Races,criterion=None, batch_size=None,epoch=0,config=None,device='cuda:0',save_output=False):
    # torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        len_test = len(raceDB.test_dog_ids)
        test_idx = range(0,len_test)
        dogs = [x for x in  raceDB.test_dogs.values()]  #[Dog]
        dog_input = [inner for inner in [x for x in raceDB.get_dog_test(test_idx)]] #[[DogInput]]
    
        X = raceDB.batches['packed_y']
        X_d = raceDB.packed_y_data
        hidden_in = torch.stack([x.hidden for x in dogs]).transpose(0,1)

        output,hidden = model((X,X_d),h=hidden_in) # Shape List[Tensor[Dog]]
        
        hidden = hidden.transpose(0,1)
        for i,dog in enumerate(dogs):
            dog.hidden_test = hidden[i]


        for i,dog in enumerate(dog_input):
                [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[i])]

        len_test = len(raceDB.test_race_ids)
        test_idx = range(0,len_test)

        race = raceDB.get_test_input(test_idx)
        t0 = time.perf_counter()
        all_price_df, loss, loss_p,loss_bfsp,loss_kl, correct, accuracy, mutual_info,log_loss, betfair_loss, loss_ohe = validate_model_pass(model,raceDB,race,criterion,test_idx,device=device,save_output=save_output)
        t1 = time.perf_counter()
        print(f"Test val pass Time: {t1-t0}")
        all_price_df.race_num = pd.to_numeric(all_price_df.race_num)
        all_price_df = all_price_df[all_price_df['prices']>1]

        all_price_df = clean_data(all_price_df)

        try:
            flat_track_df = all_price_df.groupby('track').sum(numeric_only=True).reset_index()
            flat_track_df['ROI < 30'] = flat_track_df.apply(lambda x: x['profit < 30']/x['outlay < 30'],axis =1)
            flat_track_df['ROI < 30 2'] = flat_track_df.apply(lambda x: x['profit < 30 2']/x['outlay < 30 2'],axis =1)
            flat_track_df['ROI relu'] = flat_track_df.apply(lambda x: x['profit_relu']/x['bet_relu'] if x['bet_relu']>0 else 0,axis =1).tolist()
            flat_track_df['ROI < 30 relu'] = flat_track_df.apply(lambda x: x['profit_relu<30']/x['bet_relu<30'] if x['bet_relu']>0 else 0,axis =1).tolist()
            flat_track_df['loss avg'] = flat_track_df.apply(lambda x: x['loss']/x['onehot_win'], axis=1)
            flat_track_df['accuracy'] = flat_track_df.apply(lambda x: x['correct']/x['onehot_win'], axis=1)
            # flat_track_df = flat_track_df.merge(right=flat_race_df, how='left', on='track')
        except Exception as e:
            print(e)
        # flat_track_df['win rate'] = flat_track_df['win']/flat_track_df['count']

        flat_date_df = all_price_df[['date','outlay < 30','profit < 30','bet_amount','profit','profit_relu','profit_relu<30']].groupby('date').sum().cumsum().reset_index()
        # print(all_price_df.columns)
        all_price_df['round_price'] = all_price_df['prices'].round(0)
        flat_price_df = all_price_df[['round_price','bet_amount','profit','profit_relu','bet_relu']].groupby('round_price').sum().cumsum().reset_index()
        flat_price_df['relu_roi'] = flat_price_df['profit_relu']/flat_price_df['bet_relu']
        flat_price_df['roi'] = flat_price_df['profit_relu']/flat_price_df['bet_amount']
        flat_price_df_wandb = wandb.Table(dataframe=flat_price_df.reset_index())

        flat_date_df_wandb = wandb.Table(dataframe=flat_date_df.reset_index())

        flat_date_df = all_price_df[['date','outlay < 30','profit < 30','bet_amount','profit','profit_relu<30','bet_relu<30','loss', 'onehot_win', 'correct']].groupby('date').sum().rolling(window=5).sum().reset_index()
        flat_date_df['loss'] = flat_date_df['loss']/flat_date_df['onehot_win']
        flat_date_df['accuracy'] = flat_date_df['correct']/flat_date_df['onehot_win']
        flat_date_df['roi<30_relu'] = flat_date_df['profit_relu<30']/flat_date_df['bet_relu<30']
        flat_date_df['roi<30'] = flat_date_df['profit < 30']/flat_date_df['outlay < 30']
        flat_date_df_sum_wandb = wandb.Table(dataframe=flat_date_df.reset_index())

        simple_df = wandb.Table(dataframe=all_price_df[['imp_prob','pred_prob', 'loss','onehot_win','mutual_info' ]].reset_index())

        race_df = all_price_df.groupby('raceID')[['profit','bet_amount']].sum().reset_index()
        race_df['win'] = np.where(race_df['profit']>0,1,0)
        race_df['bet_count'] =  np.where(race_df['bet_amount']>0,1,0)
        win_rate = race_df['win'].sum()/race_df['bet_count'].sum()
        win_rate_30 = all_price_df.query('prices<30')['win'].sum()/all_price_df.query('prices<30')['bet_count'].sum()  

        
        
        stats_dict = {"accuracy": correct/len_test,
                    'multibet profit':all_price_df['profit'].sum(),
                    'multibet profit < 30':all_price_df['profit < 30'].sum(),
                    'multibet profit < 30 sd':all_price_df['profit < 30'].std(),
                    'multibet outlay < 30':all_price_df['outlay < 30'].sum(),
                    'ROI < 30':all_price_df['profit < 30'].sum()/all_price_df['outlay < 30'].sum(),
                    'ROI < 30 2':all_price_df['profit < 30 2'].sum()/all_price_df['outlay < 30 2'].sum(),
                    'kelly roi':all_price_df['profit_kelly'].sum()/all_price_df['bet_amount_kelly'].sum(),
                    'kelly seq roi':all_price_df['profit_kelly_sq'].sum()/all_price_df['bet_kelly_seq'].sum(),
                    'kelly seq roi < 30':all_price_df.query('prices<30')['profit_kelly_sq'].sum()/all_price_df.query('prices<30')['bet_kelly_seq'].sum(),
                    "loss_val":torch.mean(loss).item(),
                    'loss_p':torch.mean(loss_p).item(),
                    'loss_bfsp':all_price_df['loss_bfsp'].mean(),
                    'loss_kl':all_price_df['loss'].mean(),
                    'flat_simple':all_price_df['simple'].sum(),
                    'profit_relu':all_price_df['profit_relu'].sum(),
                    'relu roi':all_price_df['profit_relu'].sum()/all_price_df['bet_relu'].sum(),
                    'mutual_info':mutual_info.mean(),
                    'epoch':epoch,
                    'favorite_accuracy':all_price_df['favorite_correct'].sum()/all_price_df['raceID'].nunique(),
                    'win_price_mean':all_price_df['win_price'].mean(),
                    'win_rate':all_price_df['win'].sum()/all_price_df['bet_count'].sum(),
                    'win_rate < 30':all_price_df['win < 30'].sum()/all_price_df['bet_count < 30'].sum(),
                    'race_win_rate':win_rate,
                    'race_win_rate_30':win_rate_30,
                    'avg_win_price':all_price_df[all_price_df['win < 30'] == 1]['prices'].mean(),
                    'model_roi':all_price_df['profit_model'].sum()/all_price_df['bet_amount_model'].sum(),
                    'model_roi<30':all_price_df.query('prices<30')['profit_model'].sum()/all_price_df.query('prices<30')['bet_amount_model'].sum(),
                    'test_log_loss':log_loss,
                    'test_betfair_loss':betfair_loss,
                    'test_loss_ohe':loss_ohe,
                    }
        
        flat_track_wandb = wandb.Table(dataframe=flat_track_df.reset_index())
        try:
            wandb.log({'flat_track':flat_track_wandb})
            wandb.log({'flat_date':flat_date_df_wandb})
            wandb.log({'flat_price_df':flat_price_df_wandb})
            wandb.log({'flat_date_sum':flat_date_df_sum_wandb})
            wandb.log({'simple_df':simple_df})
        except Exception as e:
            print(e)
        wandb.log(stats_dict)

        flat_track_df.to_csv(f'./model_all_price/{wandb.run.name} - flat_df.csv')
        # all_price_df.reset_index().to_feather(f'./model_all_price/{wandb.run.name} - all_price_df.fth')

        if epoch%100==0:
            all_price_df.reset_index().to_feather(f'./model_all_price/{wandb.run.name} - all_price_df.fth')
        
        wandb.log(stats_dict)
        wandb.log({"accuracy2": correct/len_test})

        return stats_dict

@torch.no_grad()
def validate_model_v3(model:GRUNetv3,raceDB:Races,criterion=None, batch_size=None,epoch=0,config=None,device='cuda:0',save_output=False):
    # torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)

    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():

        len_test = len(raceDB.val_dog_ids)
        val_idx = range(0,len_test)
        dogs = [x for x in  raceDB.val_dogs.values()]  #[Dog]
        dog_input = [inner for inner in [x for x in raceDB.get_dog_val(val_idx)]] #[[DogInput]]
    
        X = raceDB.batches['packed_v']
        X_d = raceDB.packed_v_data

        hidden_in = torch.stack([x.hidden_test for x in dogs]).transpose(0,1)


        output,hidden = model((X,X_d),h=hidden_in) # Shape List[Tensor[Dog]]
        hidden = hidden.transpose(0,1)
        for i,dog in enumerate(dogs):
            dog.hidden_test = hidden[i]


        for i,dog in enumerate(dog_input):
                [setattr(obj, 'hidden_out', val.detach()) for obj, val in zip(dog,output[i])]

        len_test = len(raceDB.val_race_ids)
        val_idx_races = range(0,len(raceDB.val_race_ids))
        race = raceDB.get_val_input(val_idx_races)
        t0 = time.perf_counter()
        all_price_df, loss, loss_p,loss_bfsp,loss_kl, correct, accuracy, mutual_info,log_loss, betfair_loss, loss_ohe = validate_model_pass(model,raceDB,race,criterion,val_idx_races,device=device,save_output=save_output)
        t1 = time.perf_counter()
        print(f"val val pass Time: {t1-t0}")

        all_price_df.race_num = pd.to_numeric(all_price_df.race_num)
        date = datetime.datetime.today().date()
        pred_df = all_price_df
        pred_df['date'] = pd.to_datetime(pred_df.date).dt.date
        pred_df=pred_df[pred_df['date']  == date]
        predictins = pred_df
        predictins.reset_index().to_feather(f'./model_all_price/preds{wandb.run.name} - val_all_price_df.fth')
        all_price_df = all_price_df[all_price_df['prices']>1]

        all_price_df = clean_data(all_price_df)


        try:
            flat_track_df = all_price_df.groupby('track').sum(numeric_only=True).reset_index()
            flat_track_df['ROI < 30'] = flat_track_df.apply(lambda x: x['profit < 30']/x['outlay < 30'],axis =1)
            flat_track_df['ROI < 30 2'] = flat_track_df.apply(lambda x: x['profit < 30 2']/x['outlay < 30 2'],axis =1)
            flat_track_df['ROI relu'] = flat_track_df.apply(lambda x: x['profit_relu']/x['bet_relu'] if x['bet_relu']>0 else 0,axis =1).tolist()
            flat_track_df['ROI < 30 relu'] = flat_track_df.apply(lambda x: x['profit_relu<30']/x['bet_relu<30'] if x['bet_relu']>0 else 0,axis =1).tolist()
            flat_track_df['loss avg'] = flat_track_df.apply(lambda x: x['loss']/x['onehot_win'], axis=1)
            flat_track_df['accuracy'] = flat_track_df.apply(lambda x: x['correct']/x['onehot_win'], axis=1)
            # flat_track_df = flat_track_df.merge(right=flat_race_df, how='left', on='track')
        except Exception as e:
            print(e)

        flat_date_df = all_price_df[['date','outlay < 30','profit < 30','bet_amount','profit','profit_relu','profit_relu<30',]].groupby('date').sum().cumsum().reset_index()
        all_price_df['round_price'] = all_price_df['prices'].round(0)
        flat_price_df = all_price_df[['round_price','bet_amount','profit','profit_relu','bet_relu']].groupby('round_price').sum().cumsum().reset_index()
        flat_price_df['relu_roi'] = flat_price_df['profit_relu']/flat_price_df['bet_relu']
        flat_price_df['roi'] = flat_price_df['profit_relu']/flat_price_df['bet_amount']
        flat_price_df_wandb = wandb.Table(dataframe=flat_price_df.reset_index())

        flat_date_df_wandb = wandb.Table(dataframe=flat_date_df.reset_index())

        flat_date_df = all_price_df[['date','outlay < 30','profit < 30','bet_amount','profit','profit_relu<30','bet_relu<30','loss', 'onehot_win', 'correct',]].groupby('date').sum().rolling(window=5).sum().reset_index()
        flat_date_df['loss'] = flat_date_df['loss']/flat_date_df['onehot_win']
        flat_date_df['accuracy'] = flat_date_df['correct']/flat_date_df['onehot_win']
        flat_date_df['roi<30_relu'] = flat_date_df['profit_relu<30']/flat_date_df['bet_relu<30']
        flat_date_df['roi<30'] = flat_date_df['profit < 30']/flat_date_df['outlay < 30']
        flat_date_df_sum_wandb = wandb.Table(dataframe=flat_date_df.reset_index())

        race_df = all_price_df.groupby('raceID')[['profit','bet_amount']].sum().reset_index()
        race_df['win'] = np.where(race_df['profit']>0,1,0)
        race_df['bet_count'] =  np.where(race_df['bet_amount']>0,1,0)
        win_rate = race_df['win'].sum()/race_df['bet_count'].sum()
        win_rate_30 = all_price_df.query('prices<30')['win'].sum()/all_price_df.query('prices<30')['bet_count'].sum()  
        
        stats_dict = {"val_accuracy": correct/len(raceDB.val_race_ids),
                    'val_multibet profit':all_price_df['profit'].sum(),
                    'val_kelly roi':all_price_df['profit_kelly'].sum()/all_price_df['bet_amount_kelly'].sum(),
                    'val_kelly seq roi':all_price_df['profit_kelly_sq'].sum()/all_price_df['bet_kelly_seq'].sum(),
                    'val_kelly seq roi < 30':all_price_df.query('prices<30')['profit_kelly_sq'].sum()/all_price_df.query('prices<30')['bet_kelly_seq'].sum(),
                    'val_multibet profit < 30':all_price_df['profit < 30'].sum(),
                    'val_multibet profit < 30 sd':all_price_df['profit < 30'].std(),
                    'val_multibet outlay < 30':all_price_df['outlay < 30'].sum(),
                    'val_ROI < 30':all_price_df['profit < 30'].sum()/all_price_df['outlay < 30'].sum(),
                    'val_ROI < 30 2':all_price_df['profit < 30 2'].sum()/all_price_df['outlay < 30 2'].sum(),
                    "val_loss_val":torch.mean(loss).item(),
                    'val_loss_p':torch.mean(loss_p).item(),
                    'val_loss_bfsp':all_price_df['loss_bfsp'].mean(),
                    'val_loss_kl':all_price_df['loss'].mean(),
                    'val_flat_simple':all_price_df['simple'].sum(),
                    'val_profit_relu':all_price_df['profit_relu'].sum(),
                    'val_relu roi':all_price_df['profit_relu'].sum()/all_price_df['bet_relu'].sum(),
                    'val_mutual_info':mutual_info.mean(),
                    'epoch':epoch,
                    'valfavorite_accuracy':all_price_df['favorite_correct'].sum()/all_price_df['raceID'].nunique(),
                    'val_win_price_mean':all_price_df['win_price'].mean(),
                    'val_win_rate':all_price_df['win'].sum()/all_price_df['bet_count'].sum(),
                    'val_win_rate < 30':all_price_df['win < 30'].sum()/all_price_df['bet_count < 30'].sum(),
                    'val_race_win_rate':win_rate,
                    'val_race_win_rate_30':win_rate_30, 
                    'val_avg_win_price':all_price_df[all_price_df['win < 30'] == 1]['prices'].mean(),
                    'val_model_roi':all_price_df['profit_model'].sum()/all_price_df['bet_amount_model'].sum(),
                    'val_model_roi<30':all_price_df.query('prices<30')['profit_model'].sum()/all_price_df.query('prices<30')['bet_amount_model'].sum(),
                    'val_log_loss':log_loss,
                    'val_betfair_loss':betfair_loss,
                    'val_loss_ohe':loss_ohe,
                    }
        
        # print(stats_dict)
        
        flat_track_wandb = wandb.Table(dataframe=flat_track_df.reset_index())
        try:
            wandb.log({'val_flat_track':flat_track_wandb})
            wandb.log({'val_flat_date':flat_date_df_wandb})
            wandb.log({'val_flat_price_df':flat_price_df_wandb})
            wandb.log({'val_flat_date_sum':flat_date_df_sum_wandb})
        except Exception as e:
            print(e)
        wandb.log(stats_dict)

        flat_track_df.to_csv(f'./model_all_price/{wandb.run.name} - flat_df.csv')

        # all_price_df.reset_index().to_excel(f'./model_all_price/{wandb.run.name} - val_all_price_df.xlsx')
        # all_price_df.reset_index().to_feather(f'./model_all_price/{wandb.run.name} - val_all_price_df.fth')

        if epoch%100==0:
            all_price_df.reset_index().to_feather(f'./model_all_price/{wandb.run.name} - val_all_price_df.fth')
        

        return stats_dict
