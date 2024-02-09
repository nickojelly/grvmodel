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

def train_regular_v3(model:GRUNetv3,raceDB:Races, criterion, optimizer,scheduler, config=None,update=False):
    # torch.autograd.set_detect_anomaly(True)

    last = 0
    if config!=None:
        # batch_size = config['batch_size']
        epochs = config['epochs']
    else:
        batch_size = 25
    n_layers = 2 # config['num_layers']
    len_train_dogs = len(raceDB.train_dog_ids)
    len_train_races = len(raceDB.train_race_ids)
    example_ct = 0  # number of examples seen
    batch_ct = 0
    m = nn.LogSoftmax(dim=1)

    num_batches = raceDB.batches['num_batches']

    for epoch in trange(epochs):
        model.train()
        if update:
            model.eval()

        for i in range(num_batches):
            with torch.cuda.amp.autocast():
                dogs = raceDB.batches['dogs'][i]
                train_dog_input = raceDB.batches['train_dog_input'][i]
                # train_dog_input_np = raceDB.batches['train_dog_input_np'][i]
                batch_races = raceDB.batches['batch_races'][i]
                batch_races_ids = raceDB.batches['batch_races_ids'][i]
                X = raceDB.batches['packed_x'][i]

                example_ct+=len(batch_races)

                t1 = time.perf_counter()
                hidden_in = torch.stack([x.hidden for x in dogs]).transpose(0,1)
                output,hidden = model(X, h=hidden_in)
                
                hidden = hidden.transpose(0,1)
                t2 = time.perf_counter()
                # print(f"{t2-t1=}")
                t1 = time.perf_counter()
                for i,dog in enumerate(train_dog_input):
                    [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[i])]
                t3 = time.perf_counter()
                # print(f"{t3-t2=}")
                [setattr(obj, 'hidden', val) for obj, val in zip(dogs,hidden)]

                [setattr(race, 'hidden_in', torch.cat([race.race_dist]+[race.race_track]+[d.hidden_out for d in race.dogs])) for race in batch_races]

                race = batch_races

                X2 = torch.stack([r.hidden_in for r in race]) #Input for FFNN
                y = torch.stack([x.classes for x in race])
                w = torch.stack([x.new_win_weight for x in race])
                # w = torch.stack([x.win_price_weightv2  for x in race])

                output,_ = model(X2, p1=False)
                model.zero_grad(set_to_none=True)

                # print(f"{output=}\n{y=},{w=}")

                epoch_loss = criterion(output, y)*w

                # print(epoch_loss)
            t6 = time.perf_counter()
            if not update:
                epoch_loss.mean().backward()
                optimizer.step()
                wandb.log({"loss_1": torch.mean(epoch_loss).item(), 'epoch':epoch}, step = example_ct)
            raceDB.detach_hidden(dogs)
            t7 = time.perf_counter()


        if (epoch)%20==0:
            t8 = time.perf_counter()
            raceDB.reset_hidden(num_layers=2, hidden_size=128)
            test_model_v3(model,raceDB, criterion=criterion, epoch=epoch)
            validate_model_v3(model,raceDB, criterion=criterion, epoch=epoch)
            t9 = time.perf_counter()
        if (epoch)%100==0:
            raceDB.create_hidden_states_dict_v2()
            model_saver_wandb(model, optimizer, epoch, 0.1, raceDB.hidden_states_dict_gru_v6, raceDB.train_hidden_dict, model_name="long nsw new  22000 RUN")
            if update:
                break
        if not update:
            #print('reset hidden')
            raceDB.reset_hidden(num_layers=2, hidden_size=128)
        torch.cuda.empty_cache()

    return model

# @torch.amp.autocast(device_type='cuda')
def train_double_v3(model:GRUNetv4_stacking,raceDB:Races, criterion:nn.CrossEntropyLoss, optimizer,scheduler, config=None,update=False):

    example_ct = 0  # number of examples seen
    num_batches = raceDB.batches['num_batches']
    epochs = config['epochs']

    for epoch in trange(epochs):
        model.train()
        hidden_state_init = model.model_list[0].h0
        raceDB.reset_hidden_w_param(hidden_state_init,num_layers=2, hidden_size=config['hidden_size'])
        
        raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(config['hidden_size']+config['f1_layer_size'])).to('cuda:0')
        epoch_loss = []
        epoch_loss_p = []
        for i in range(num_batches-1):
            # print(i)
            with torch.cuda.amp.autocast():
                simple_model = model.model_list[i]
                # raceDB.reset_hidden_w_param(simple_model.h0,num_layers=2, hidden_size=config['hidden_size'])
                dogs = raceDB.batches['dogs'][i]
                train_dog_input = raceDB.batches['train_dog_input'][i]
                # train_dog_input_np = raceDB.batches['train_dog_input_np'][i]
                batch_races = raceDB.batches['batch_races'][i]
                batch_races_ids = raceDB.batches['batch_races_ids'][i]
                X = raceDB.batches['packed_x'][i]
                X_d = raceDB.packed_x_data[i]

                example_ct+=len(batch_races)
                hidden_in = torch.stack([x.hidden for x in dogs]).transpose(0,1)
                output,hidden = simple_model((X,X_d), h=hidden_in)
                
                hidden = hidden.transpose(0,1)


                for j,dog in enumerate(train_dog_input):
                    [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[j])]

                # [setattr(obj, 'hidden', val) for obj, val in zip(dogs,hidden)]

                [setattr(race, 'hidden_in', torch.cat([race.race_dist]+[race.race_track]+[d.hidden_out for d in race.dogs])) for race in batch_races]

                race = batch_races

                X2 = torch.stack([r.hidden_in for r in race]) #Input for FFNN
                y = torch.stack([x.classes for x in race])
                y_ohe = torch.stack([x.one_hot_class for x in race])
                y_p = torch.stack([x.prob for x in race])
                w = torch.stack([x.new_win_weight for x in race])
                # w = torch.stack([x.win_price_weightv2  for x in race])

                output,_,output_p = simple_model(X2, p1=False)
                loss = criterion(output, y)*w
                #loss_ohe = criterion(output, y_ohe)*w
                
                loss_p = criterion(output_p, y_p)*w

                epoch_loss.append(loss.mean())
                epoch_loss_p.append(loss_p.mean())
                wandb.log({f"loss_{i}": torch.mean(loss.mean()).item(), 'epoch':epoch})
                simple_model.zero_grad(set_to_none=True)


                # print(epoch_loss)
                # raceDB.reset_hidden_w_param(hidden_state_init,num_layers=2, hidden_size=config['hidden_size'])
                raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(256+64)).to('cuda:0')
        t6 = time.perf_counter()
        if not update:
            # print(epoch_loss)
            epoch_loss = torch.stack(epoch_loss)
            epoch_loss_p = torch.stack(epoch_loss_p)
            optimizer.zero_grad()
            (epoch_loss_p+epoch_loss).mean().backward()
            optimizer.step()
            wandb.log({f"loss_avg": torch.mean(epoch_loss).item(), 'epoch':epoch})

        #Ensemble
        with torch.cuda.amp.autocast():
            dogs = raceDB.batches['dogs'][-1]
            train_dog_input = raceDB.batches['train_dog_input'][-1]
            batch_races = raceDB.batches['batch_races'][-1]
            batch_races_ids = raceDB.batches['batch_races_ids'][-1]
            X = raceDB.batches['packed_x'][-1]
            X_d = raceDB.packed_x_data[-1]

            model_losses = []
            for i in range(num_batches-1):
                # print(i)
                simple_model = model.model_list[i]
                output,hidden = simple_model((X,X_d), h=hidden_in)

                hidden = hidden.transpose(0,1)


                for j,dog in enumerate(train_dog_input):
                    [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[j])]

                # [setattr(obj, 'hidden', val) for obj, val in zip(dogs,hidden)]

                [setattr(race, 'hidden_in', torch.cat([race.race_dist]+[race.race_track]+[d.hidden_out for d in race.dogs])) for race in batch_races]

                race = batch_races

                X2 = torch.stack([r.hidden_in for r in race]) #Input for FFNN
                y = torch.stack([x.classes for x in race])
                y_ohe = torch.stack([x.one_hot_class for x in race])
                y_p = torch.stack([x.prob for x in race])
                w = torch.stack([x.new_win_weight for x in race])
                # w = torch.stack([x.win_price_weightv2  for x in race])

                output,_,output_p = simple_model(X2, p1=False)
                epoch_loss = criterion(output, y).mean()
                model_losses.append(epoch_loss)
            
            output = model(stacking=True)
            epoch_loss = criterion(output, y).mean()
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()
            wandb.log({f"loss_ensemble": torch.mean(epoch_loss).item(),'loss_reg': torch.stack(model_losses).mean(), 'epoch':epoch})

        if (epoch)%3==0:
            for i in range(num_batches-1):
                simple_model = model.model_list[i]
                test_model_v3(simple_model,raceDB, criterion=criterion, epoch=epoch)
            test_model_v3(model,raceDB, criterion=criterion, epoch=epoch,ensemble=True)


        # if (epoch)%3==0:
        #     t8 = time.perf_counter()
        #     # raceDB.reset_hidden_w_param(hidden_state_init,num_layers=2, hidden_size=config['hidden_size'])
        #     # raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(256+64)).to('cuda:0')
        #     test_model_v3(model,raceDB, criterion=criterion, epoch=epoch)
        #     # raceDB.reset_hidden_w_param(hidden_state_init,num_layers=2, hidden_size=config['hidden_size'])
        #     # raceDB.reset_hidden_w_param(hidden_state_init,num_layers=2, hidden_size=config['hidden_size'])
        #     # raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(256+64)).to('cuda:0')
        #     validate_model_v3(model,raceDB, criterion=criterion, epoch=epoch)
        #     t9 = time.perf_counter()
        # if (epoch)%20==0:
        #     raceDB.create_hidden_states_dict_v2()
        #     model_saver_wandb(model, optimizer, epoch, 0.1, raceDB.hidden_states_dict_gru_v6, raceDB.train_hidden_dict, model_name="long nsw new  22000 RUN")
        #     if update:
        #         break
        # if not update:
        #     #print('reset hidden')
        #     # raceDB.reset_hidden(num_layers=2, hidden_size=config['hidden_size'])    
        #     hidden_state_init = model.h0
        #     raceDB.reset_hidden_w_param(hidden_state_init,num_layers=2, hidden_size=config['hidden_size'])
        # torch.cuda.empty_cache()

    return model


def clean_data(df):
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
    df['win_price'] = np.where(df['onehot_win'], df['prices'], 0)

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

    # df['bet_relu'] = df['bet_amount'] * 1/df['mutual_info']
    # df['profit_relu'] = np.where(
    #     df['onehot_win'],
    #     df['bet_relu'] * (df['prices'] - 1) * 0.95,
    #     -1 * df['bet_relu']
    # )

    # df['bet_relu<30'] = np.where(df['prices'] < 20, df['bet_amount'] * df['relu'], 0)
    # df['profit_relu<30'] = np.where(df['prices'] < 20, df['profit_relu'], 0)

    

    return df

import sys

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            # print(m.__class__.__name__)
            m.train()

def get_monte_carlo_predictions(data,
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
            output,relu,output_p,= model(data, p1=False)
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


def validate_model_pass(model:GRUNetv3,raceDB:Races,race, criterion,test_idx,device='cuda:0',ensemble = False):

    sft_max = nn.Softmax(dim=-1)
    Xt = torch.stack([r.hidden_in for r in race]) #Input for FFNN
    y = torch.stack([x.classes for x in race])
    y_p = torch.stack([x.prob for x in race])
    y_bfsp = torch.stack([x.implied_prob for x in race])    
    r = torch.stack([r.relu for r in race])

    if ensemble:
        output= model()
        output_p = output
    else:
        output,relu,output_p,= model(Xt, p1=False)
        mean, variance, entropy, mutual_info = get_monte_carlo_predictions(Xt, 100,model,8,y.shape[0])
    # print(f"{mean=}\n{variance=}\n{entropy=}\n{mutual_info=}")
    
    relu = output_p.mean(dim=1)

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
    softmax_preds1 = sft_max(output/0.5)
    softmax_preds2 = sft_max(output/0.5)

    loss = criterion(output, y).mean()
    loss_p = criterion(output_p, y_p).mean()
    loss_bfsp = criterion(y_bfsp, y).mean()

    loss_tensor = validation_CLE(output,y)
    loss_tensor_bfsp = validation_CLE(y_bfsp,y)

    test_races = race

    price_tensor = torch.tensor([x.prices for x in test_races], device=device)

    profit_tensor = price_tensor*correct_tensor*0.95-pred_label

    races = {}
    races['raw_margins'] = [x.raw_margins for x in test_races]
    races['correct'] = correct_tensor.tolist()
    races['simple']  = profit_tensor.tolist()
    # races['output'] = [x.margins for x in test_races]
    races['relu'] = [[x]*8 for x in relu.cpu().tolist()]
    # print(races['relu'])
    races['pred_prob'] = softmax_preds.tolist()
    races['pred_prob2'] = softmax_preds2.tolist()
    races['prices'] = [x.prices for x in test_races]
    races['imp_prob'] = [x.implied_prob.tolist()  for x in test_races]
    races['pred_price'] = (1/softmax_preds).tolist()
    races['pred_price1'] = (1/softmax_preds1).tolist()
    races['pred_price2'] = (1/softmax_preds2).tolist()
    # races['pred_prob'] = [x.tolist() for x in races['pred_prob']]
    races['classes'] = [x.classes.tolist() for x in test_races]
    races['track'] = [x.track_name for x in test_races]
    races['onehot_win'] = [x.one_hot_class.tolist() for x in test_races]
    # races['date'] = [x.race_date for x in test_races]
    races['dogID'] = [x.list_dog_ids() for x in race]
    races['dog_name'] = [x.list_dog_names() for x in race]
    races['raceID'] = [[x.raceid]*8 for x in race]
    races['date'] = [[x.race_date]*8 for x in race]
    # races['entropy'] = [[x]*8 for x in entropy.tolist()]
    # races['mutual_info'] = [[x]*8 for x in mutual_info.tolist()]
    races['race_num'] = [[int(x.race_num)]*8 for x in race]
    races['loss'] = loss_tensor.tolist()
    races['loss_bfsp'] = loss_tensor_bfsp.tolist()
    races['correct'] = correct_tensor.tolist()
    races['favorite_correct'] = favorite_tensor.tolist()


    accuracy = correct/len(predicted)

    # for k,v in races.items():
    #     print(f"{k} length is {len(v)} type {type(v[0])}")

    races['one_hot_win'] = onehot_win.tolist()
    races['track'] = [[x.track_name]*8 for x in test_races]

    for k,value in races.items():
        # print(k)
        races[k] = [item for sublist in value for item in sublist]
        # print(len(races[k]))
    all_price_df = pd.DataFrame(races)

    return all_price_df, loss, loss_p,loss_bfsp, correct, accuracy,0

#Testing
@torch.no_grad()
def test_model_v3(model:GRUNetv4_extra,raceDB:Races,criterion=None, batch_size=None,epoch=10,config=None,device='cuda:0',ensemble=False):
    # torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)
    if ensemble:
        model_num = 'ensemble'
    else:
        model_num = model.model_number

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        len_test = len(raceDB.test_dog_ids)
        test_idx = range(0,len_test)
        race = raceDB.get_test_input(test_idx)

        if ensemble:
            all_price_df, loss, loss_p,loss_bfsp, correct, accuracy, mutual_info = validate_model_pass(model,raceDB,race,criterion,test_idx,device=device,ensemble=True)
        else:
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


            raceDB.margin_from_dog_to_race_v3(mode='test')

            len_test = len(raceDB.test_race_ids)
            test_idx = range(0,len_test)

            

            all_price_df, loss, loss_p,loss_bfsp, correct, accuracy, mutual_info = validate_model_pass(model,raceDB,race,criterion,test_idx,device=device)

        all_price_df.race_num = pd.to_numeric(all_price_df.race_num)

        all_price_df = all_price_df[all_price_df['prices']>1]

        all_price_df = clean_data(all_price_df)

        # try:
        #     flat_track_df = all_price_df.groupby('track').sum(numeric_only=True).reset_index()
        #     flat_track_df['ROI < 30'] = flat_track_df.apply(lambda x: x['profit < 30']/x['outlay < 30'],axis =1)
        #     flat_track_df['ROI < 30 2'] = flat_track_df.apply(lambda x: x['profit < 30 2']/x['outlay < 30 2'],axis =1)
        #     flat_track_df['ROI relu'] = flat_track_df.apply(lambda x: x['profit_relu']/x['bet_relu'] if x['bet_relu']>0 else 0,axis =1).tolist()
        #     flat_track_df['ROI < 30 relu'] = flat_track_df.apply(lambda x: x['profit_relu<30']/x['bet_relu<30'] if x['bet_relu']>0 else 0,axis =1).tolist()
        #     flat_track_df['loss avg'] = flat_track_df.apply(lambda x: x['loss']/x['onehot_win'], axis=1)
        #     flat_track_df['accuracy'] = flat_track_df.apply(lambda x: x['correct']/x['onehot_win'], axis=1)
        #     # flat_track_df = flat_track_df.merge(right=flat_race_df, how='left', on='track')
        # except Exception as e:
        #     print(e)
        # # flat_track_df['win rate'] = flat_track_df['win']/flat_track_df['count']

        # flat_date_df = all_price_df[['date','outlay < 30','profit < 30','bet_amount','profit','profit_relu','profit_relu<30']].groupby('date').sum().cumsum().reset_index()
        # # print(all_price_df.columns)
        # all_price_df['round_price'] = all_price_df['prices'].round(0)
        # flat_price_df = all_price_df[['round_price','bet_amount','profit','profit_relu','bet_relu']].groupby('round_price').sum().cumsum().reset_index()
        # flat_price_df['relu_roi'] = flat_price_df['profit_relu']/flat_price_df['bet_relu']
        # flat_price_df_wandb = wandb.Table(dataframe=flat_price_df.reset_index())

        # flat_date_df_wandb = wandb.Table(dataframe=flat_date_df.reset_index())

        # flat_date_df = all_price_df[['date','outlay < 30','profit < 30','bet_amount','profit','profit_relu<30','bet_relu<30','loss', 'onehot_win', 'correct']].groupby('date').sum().rolling(window=5).sum().reset_index()
        # flat_date_df['loss'] = flat_date_df['loss']/flat_date_df['onehot_win']
        # flat_date_df['accuracy'] = flat_date_df['correct']/flat_date_df['onehot_win']
        # flat_date_df['roi<30_relu'] = flat_date_df['profit_relu<30']/flat_date_df['bet_relu<30']
        # flat_date_df['roi<30'] = flat_date_df['profit < 30']/flat_date_df['outlay < 30']
        # flat_date_df_sum_wandb = wandb.Table(dataframe=flat_date_df.reset_index())

        # simple_df = wandb.Table(dataframe=all_price_df[['imp_prob','pred_prob', 'loss','onehot_win' ]].reset_index())

        
        # print(correct)
        # print( correct/len(raceDB.test_race_ids))
        # print(accuracy)
        stats_dict = {F"accuracy_{model_num}": accuracy,
                    # 'multibet profit':all_price_df['profit'].sum(),
                    # 'multibet profit < 30':all_price_df['profit < 30'].sum(),
                    # 'multibet profit < 30 sd':all_price_df['profit < 30'].std(),
                    # 'multibet outlay < 30':all_price_df['outlay < 30'].sum(),
                    f'ROI < 30_{model_num}':all_price_df['profit < 30'].sum()/all_price_df['outlay < 30'].sum(),
                    # 'ROI < 30 2':all_price_df['profit < 30 2'].sum()/all_price_df['outlay < 30 2'].sum(),
                    # 'kelly roi':all_price_df['profit_kelly'].sum()/all_price_df['bet_amount_kelly'].sum(),
                    f"loss_val_{model_num}":torch.mean(loss).item(),
                    # 'loss_p':torch.mean(loss_p).item(),
                    # 'loss_bfsp':torch.mean(loss_bfsp).item(),
                    # 'flat_simple':all_price_df['simple'].sum(),
                    # 'profit_relu':all_price_df['profit_relu'].sum(),
                    # 'relu roi':all_price_df['profit_relu'].sum()/all_price_df['bet_relu'].sum(),
                    # 'mutual_info':mutual_info.mean(),
                    # 'epoch':epoch,
                    # 'favorite_accuracy':all_price_df['favorite_correct'].sum()/all_price_df['raceID'].nunique(),
                    }
        
        # flat_track_wandb = wandb.Table(dataframe=flat_track_df.reset_index())
        # try:
        #     pass
        #     # wandb.log({'flat_track':flat_track_wandb})
        #     # wandb.log({'flat_date':flat_date_df_wandb})
        #     # wandb.log({'flat_price_df':flat_price_df_wandb})
        #     # wandb.log({'flat_date_sum':flat_date_df_sum_wandb})
        #     # wandb.log({'simple_df':simple_df})
        # except Exception as e:
        #     print(e)
        wandb.log(stats_dict)

        # flat_track_df.to_csv(f'./model_all_price/{wandb.run.name} - flat_df.csv')

        # if epoch%100==0:
        #     all_price_df.reset_index().to_feather(f'./model_all_price/{wandb.run.name} - all_price_df.fth')
        
        # wandb.log(stats_dict)
        # wandb.log({"accuracy2": correct/len_test})

        return accuracy

@torch.no_grad()
def validate_model_v3(model:GRUNetv3,raceDB:Races,criterion=None, batch_size=None,epoch=10,config=None,device='cuda:0'):
    # torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)

    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():

        len_test = len(raceDB.val_dog_ids)
        val_idx = range(0,len_test)

        # dog_inputs = [[z.full_input for z in inner] for inner in [x for x in raceDB.train_dogs.values()]]
        # Uses train dogs here because only dogs in Train set will have new hidden values?? Does that make sense?
        # no will jumble around but wont throw error in feeding model, because when building dogs sorted, 
        # lengths will be made to match size of X(test)
        # Shouldn't have much of an effect due to size of test set??
        dogs = [x for x in  raceDB.val_dogs.values()]  #[Dog]
        dog_input = [inner for inner in [x for x in raceDB.get_dog_val(val_idx)]] #[[DogInput]]
    
        X = raceDB.batches['packed_v']
        X_d = raceDB.packed_v_data

        # train = [torch.stack(n,0) for n in [[z.stats for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]]
        # X = pack_sequence(train, enforce_sorted=False).to('cuda:0')
        # dogs_sorted = [dogs[x] for x in X[3]]

        hidden_in = torch.stack([x.hidden_test for x in dogs]).transpose(0,1)


        output,hidden = model((X,X_d),h=hidden_in) # Shape List[Tensor[Dog]]
        hidden = hidden.transpose(0,1)
        for i,dog in enumerate(dogs):
            dog.hidden_test = hidden[i]


        for i,dog in enumerate(dog_input):
                [setattr(obj, 'hidden_out', val.detach()) for obj, val in zip(dog,output[i])]


        raceDB.margin_from_dog_to_race_v3(mode='val')

        # print(f"{val_idx=}, {len_test=}")

        len_test = len(raceDB.val_race_ids)
        val_idx_races = range(0,len(raceDB.val_race_ids))
        race = raceDB.get_val_input(val_idx_races)

        all_price_df, loss, loss_p,loss_bfsp, correct, accuracy, mutual_info = validate_model_pass(model,raceDB,race,criterion,val_idx_races,device=device)


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

        flat_date_df = all_price_df[['date','outlay < 30','profit < 30','bet_amount','profit','profit_relu','profit_relu<30']].groupby('date').sum().cumsum().reset_index()
        all_price_df['round_price'] = all_price_df['prices'].round(0)
        flat_price_df = all_price_df[['round_price','bet_amount','profit','profit_relu','bet_relu']].groupby('round_price').sum().cumsum().reset_index()
        flat_price_df['relu_roi'] = flat_price_df['profit_relu']/flat_price_df['bet_relu']
        flat_price_df_wandb = wandb.Table(dataframe=flat_price_df.reset_index())

        flat_date_df_wandb = wandb.Table(dataframe=flat_date_df.reset_index())

        flat_date_df = all_price_df[['date','outlay < 30','profit < 30','bet_amount','profit','profit_relu<30','bet_relu<30','loss', 'onehot_win', 'correct']].groupby('date').sum().rolling(window=5).sum().reset_index()
        flat_date_df['loss'] = flat_date_df['loss']/flat_date_df['onehot_win']
        flat_date_df['accuracy'] = flat_date_df['correct']/flat_date_df['onehot_win']
        flat_date_df['roi<30_relu'] = flat_date_df['profit_relu<30']/flat_date_df['bet_relu<30']
        flat_date_df['roi<30'] = flat_date_df['profit < 30']/flat_date_df['outlay < 30']
        flat_date_df_sum_wandb = wandb.Table(dataframe=flat_date_df.reset_index())
        
        stats_dict = {"val_accuracy": correct/len(raceDB.val_race_ids),
                    'val_multibet profit':all_price_df['profit'].sum(),
                    'val_kelly roi':all_price_df['profit_kelly'].sum()/all_price_df['bet_amount_kelly'].sum(),
                    'val_multibet profit < 30':all_price_df['profit < 30'].sum(),
                    'val_multibet profit < 30 sd':all_price_df['profit < 30'].std(),
                    'val_multibet outlay < 30':all_price_df['outlay < 30'].sum(),
                    'val_ROI < 30':all_price_df['profit < 30'].sum()/all_price_df['outlay < 30'].sum(),
                    'val_ROI < 30 2':all_price_df['profit < 30 2'].sum()/all_price_df['outlay < 30 2'].sum(),
                    "val_loss_val":torch.mean(loss).item(),
                    'val_loss_p':torch.mean(loss_p).item(),
                    'val_loss_bfsp':torch.mean(loss_bfsp).item(),
                    'val_flat_simple':all_price_df['simple'].sum(),
                    'val_profit_relu':all_price_df['profit_relu'].sum(),
                    'val_relu roi':all_price_df['profit_relu'].sum()/all_price_df['bet_relu'].sum(),
                    'val_mutual_info':mutual_info.mean(),
                    'epoch':epoch,
                    'valfavorite_accuracy':all_price_df['favorite_correct'].sum()/all_price_df['raceID'].nunique(),
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

        all_price_df.reset_index().to_excel(f'./model_all_price/{wandb.run.name} - all_price_df.xlsx')

        if epoch%100==0:
            all_price_df.reset_index().to_feather(f'./model_all_price/{wandb.run.name} - all_price_df.fth')
        
        wandb.log(stats_dict)
        wandb.log({"accuracy2": correct/len_test})

        return accuracy
