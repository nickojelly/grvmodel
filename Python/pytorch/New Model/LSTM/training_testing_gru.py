import os
import pickle

import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
import torch
import torch.optim as optim
import wandb
from rnn_classes import *
import time
import numpy as np
from model_saver import model_saver_wandb

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

                for i,dog in enumerate(train_dog_input):
                    [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[i])]
                t3 = time.perf_counter()
                # print(f"{t3-t2=}")
                [setattr(obj, 'hidden', val) for obj, val in zip(dogs,hidden)]
                t4 = time.perf_counter()
                # print(f"{t4-t3=}")
                [setattr(race, 'hidden_in', torch.cat([race.race_dist]+[race.race_track]+[d.hidden_out for d in race.dogs])) for race in batch_races]
                t5 = time.perf_counter()
                # print(f"{t5-t4=}")
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

def train_double_v3(model:GRUNetv3,raceDB:Races, criterion, optimizer,scheduler, config=None,update=False):
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
                hidden_in = torch.stack([x.hidden for x in dogs]).transpose(0,1)
                output,hidden = model(X, h=hidden_in)
                
                hidden = hidden.transpose(0,1)


                for i,dog in enumerate(train_dog_input):
                    [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[i])]

                [setattr(obj, 'hidden', val) for obj, val in zip(dogs,hidden)]

                [setattr(race, 'hidden_in', torch.cat([race.race_dist]+[race.race_track]+[d.hidden_out for d in race.dogs])) for race in batch_races]

                race = batch_races

                X2 = torch.stack([r.hidden_in for r in race]) #Input for FFNN
                y = torch.stack([x.classes for x in race])
                y_p = torch.stack([x.prob for x in race])
                w_1 = torch.stack([x.new_win_weight for x in race])
                w_2 = torch.stack([x.win_price_weightv2  for x in race])

                output,_,output_p = model(X2, p1=False)
                model.zero_grad(set_to_none=True)

                # print(f"{output=}\n{y=},{w=}")

                epoch_loss = criterion(output, y)*w_1
                epoch_loss_p = criterion(output_p, y_p)*w_2

                # print(epoch_loss)
            t6 = time.perf_counter()
            if not update:
                (epoch_loss+3*epoch_loss_p).mean().backward()
                optimizer.step()
                wandb.log({"loss_1": torch.mean(epoch_loss).item(), 'epoch':epoch}, step = example_ct)
            raceDB.detach_hidden(dogs)
            t7 = time.perf_counter()


        if (epoch)%20==0:
            t8 = time.perf_counter()
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
        -1 * ['bet_amount_kelly'] 
    )
    df['profit2'] = np.where(
        df['onehot_win'],
        df['bet_amount2'] * (df['prices'] - 1) * 0.95,
        -1 * df['bet_amount2']
    )
    df['profit < 30 2'] = np.where(df['prices'] < 30, df['profit2'], 0)
    df['outlay < 30 2'] = np.where(df['prices'] < 30, df['bet_amount2'], 0)

    df['bet_relu'] = df['bet_amount'] * df['relu']
    df['profit_relu'] = np.where(
        df['onehot_win'],
        df['bet_relu'] * (df['prices'] - 1) * 0.95,
        -1 * df['bet_relu']
    )

    df['bet_relu<30'] = np.where(df['prices'] < 20, df['bet_amount'] * df['relu'], 0)
    df['profit_relu<30'] = np.where(df['prices'] < 20, df['profit_relu'], 0)
    

    

    return df


#Testing
@torch.no_grad()
def test_model_v3(model:GRUNetv3,raceDB:Races,criterion=None, batch_size=None,epoch=10,config=None,device='cuda:0'):
    # torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)

    correct = 0
    total = 0
    model.eval()

    price_dict = {}
    price_dict['prices'] = []
    price_dict['imp_prob'] = []
    price_dict['pred_prob'] = []
    price_dict['pred_price'] = []
    price_dict['margin'] = []
    price_dict['onehot_win'] = []
    price_dict['raceID'] = []
    price_dict['dogID'] = []
    price_dict['track'] = []
    price_dict['date'] = []
    price_dict['grade'] = []
    price_dict['loss'] = []
    race_ids = []
    # criterion = nn.CrossEntropyLoss(reduction='none')


    model.eval()


    with torch.no_grad():

        len_test = len(raceDB.test_dog_ids)
        test_idx = range(0,len_test)

        # dog_inputs = [[z.full_input for z in inner] for inner in [x for x in raceDB.train_dogs.values()]]
        # Uses train dogs here because only dogs in Train set will have new hidden values?? Does that make sense?
        # no will jumble around but wont throw error in feeding model, because when building dogs sorted, 
        # lengths will be made to match size of X(test)
        # Shouldn't have much of an effect due to size of test set??
        dogs = [x for x in  raceDB.test_dogs.values()]  #[Dog]
        dog_input = [inner for inner in [x for x in raceDB.get_dog_test(test_idx)]] #[[DogInput]]
    
        X = raceDB.batches['packed_y']

        # train = [torch.stack(n,0) for n in [[z.stats for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]]
        # X = pack_sequence(train, enforce_sorted=False).to('cuda:0')

        # dogs_sorted = [dogs[x] for x in X[3]]
        hidden_in = torch.stack([x.hidden for x in dogs]).transpose(0,1)


        output,hidden = model(X,h=hidden_in) # Shape List[Tensor[Dog]]
        hidden = hidden.transpose(0,1)
        for i,dog in enumerate(dogs):
            dog.hidden_test = hidden[i]


        for i,dog in enumerate(dog_input):
                [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[i])]


        raceDB.margin_from_dog_to_race_v3(mode='test')

        len_test = len(raceDB.test_race_ids)
        test_idx = range(0,len_test)

        race = raceDB.get_test_input(test_idx)

        Xt = torch.stack([r.hidden_in for r in race]) #Input for FFNN
        y = torch.stack([x.classes for x in race])

        r = torch.stack([r.relu for r in race])

        print(f"{len_test=}")
        
        print(r.shape)
        
        r = torch.stack([r.relu for r in race])

        print(r.shape)
        output,relu = model(Xt, p1=False)

        relu = r.sum(dim=1)
        # print(relu.shape)
        # print(relu)
        # print("here")
        # print(output)

        _, actual = torch.max(y.data, 1)
        onehot_win = F.one_hot(actual, num_classes=8)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == actual).sum().item()

        #One hot wins
        label = torch.zeros_like(y.data).scatter_(1, torch.argmax(y.data, dim=1).unsqueeze(1), 1.)
        pred_label = torch.zeros_like(output.data).scatter_(1, torch.argmax(output.data, dim=1).unsqueeze(1), 1.)
        correct_tensor = label*pred_label

        correct_l = predicted == actual

        softmax_preds = sft_max(output)
        softmax_preds1 = sft_max(output/0.5)
        softmax_preds2 = sft_max(output/0.5)

        loss = criterion(output, y).mean()

        loss_tensor = validation_CLE(output,y)

        test_races = raceDB.get_test_input(test_idx)

        price_tensor = torch.tensor([x.prices for x in test_races], device=device)

        profit_tensor = price_tensor*correct_tensor*0.95-pred_label

        # relu = torch.mean(abs(relu),dim=-1)

        # print(relu)

        races = {}
        # races['race_ids'] = [x.raceid for x in test_races]

        # for r in race[race_ids]:
        #     if len(r)>

        races['raw_margins'] = [x.raw_margins for x in test_races]
        races['correct'] = correct_tensor.tolist()
        races['simple']  = profit_tensor.tolist()
        # races['output'] = [x.margins for x in test_races]
        races['relu'] = [[x]*8 for x in relu.cpu().tolist()]
        # print(races['relu'])
        races['pred_prob'] = softmax_preds.tolist()
        races['pred_prob2'] = softmax_preds2.tolist()
        races['prices'] = [x.prices for x in test_races]
        races['imp_prob'] = [x.implied_prob  for x in test_races]
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
        races['race_num'] = [[int(x.race_num)]*8 for x in race]
        races['loss'] = loss_tensor.tolist()
        races['correct'] = correct_tensor.tolist()


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
        flat_price_df_wandb = wandb.Table(dataframe=flat_price_df.reset_index())

        flat_date_df_wandb = wandb.Table(dataframe=flat_date_df.reset_index())

        flat_date_df = all_price_df[['date','outlay < 30','profit < 30','bet_amount','profit','profit_relu<30','bet_relu<30','loss', 'onehot_win', 'correct']].groupby('date').sum().rolling(window=5).sum().reset_index()
        flat_date_df['loss'] = flat_date_df['loss']/flat_date_df['onehot_win']
        flat_date_df['accuracy'] = flat_date_df['correct']/flat_date_df['onehot_win']
        flat_date_df['roi<30_relu'] = flat_date_df['profit_relu<30']/flat_date_df['bet_relu<30']
        flat_date_df['roi<30'] = flat_date_df['profit < 30']/flat_date_df['outlay < 30']
        flat_date_df_sum_wandb = wandb.Table(dataframe=flat_date_df.reset_index())
        
        stats_dict = {"accuracy": correct/len_test,
                    'multibet profit':all_price_df['profit'].sum(),
                    'multibet profit < 30':all_price_df['profit < 30'].sum(),
                    'multibet profit < 30 sd':all_price_df['profit < 30'].std(),
                    'multibet outlay < 30':all_price_df['outlay < 30'].sum(),
                    'ROI < 30':all_price_df['profit < 30'].sum()/all_price_df['outlay < 30'].sum(),
                    'ROI < 30 2':all_price_df['profit < 30 2'].sum()/all_price_df['outlay < 30 2'].sum(),
                    'kelly roi':all_price_df['profit_kelly'].sum()/all_price_df['bet_amount_kelly'].sum(),
                    "loss_val":torch.mean(loss).item(),
                    'flat_simple':all_price_df['simple'].sum(),
                    'profit_relu':all_price_df['profit_relu'].sum(),
                    'relu roi':all_price_df['profit_relu'].sum()/all_price_df['bet_relu'].sum(),
                    'epoch':epoch,
                    }
        
        flat_track_wandb = wandb.Table(dataframe=flat_track_df.reset_index())
        try:
            wandb.log({'flat_track':flat_track_wandb})
            wandb.log({'flat_date':flat_date_df_wandb})
            wandb.log({'flat_price_df':flat_price_df_wandb})
            wandb.log({'flat_date_sum':flat_date_df_sum_wandb})
        except Exception as e:
            print(e)
        wandb.log(stats_dict)

        flat_track_df.to_csv(f'./model_all_price/{wandb.run.name} - flat_df.csv')

        if epoch%100==0:
            all_price_df.reset_index().to_feather(f'./model_all_price/{wandb.run.name} - all_price_df.fth')
        
        wandb.log(stats_dict)
        wandb.log({"accuracy2": correct/len_test})

        return accuracy

@torch.no_grad()
def validate_model_v3(model:GRUNetv3,raceDB:Races,criterion=None, batch_size=None,epoch=10,config=None,device='cuda:0'):
    # torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)

    correct = 0
    total = 0
    model.eval()

    price_dict = {}
    price_dict['prices'] = []
    price_dict['imp_prob'] = []
    price_dict['pred_prob'] = []
    price_dict['pred_price'] = []
    price_dict['margin'] = []
    price_dict['onehot_win'] = []
    price_dict['raceID'] = []
    price_dict['dogID'] = []
    price_dict['track'] = []
    price_dict['date'] = []
    price_dict['grade'] = []
    price_dict['loss'] = []
    race_ids = []
    # criterion = nn.CrossEntropyLoss(reduction='none')


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

        # train = [torch.stack(n,0) for n in [[z.stats for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]]
        # X = pack_sequence(train, enforce_sorted=False).to('cuda:0')

        # dogs_sorted = [dogs[x] for x in X[3]]
        hidden_in = torch.stack([x.hidden_test for x in dogs]).transpose(0,1)


        output,hidden = model(X,h=hidden_in) # Shape List[Tensor[Dog]]
        hidden = hidden.transpose(0,1)
        for i,dog in enumerate(dogs):
            dog.hidden_test = hidden[i]


        for i,dog in enumerate(dog_input):
                [setattr(obj, 'hidden_out', val.detach()) for obj, val in zip(dog,output[i])]


        raceDB.margin_from_dog_to_race_v3(mode='val')

        print(f"{val_idx=}, {len_test=}")


        val_idx_races = range(0,len(raceDB.val_race_ids))
        race = raceDB.get_val_input(val_idx_races)

        Xt = torch.stack([r.hidden_in for r in race]) #Input for FFNN
        y = torch.stack([x.classes for x in race])
        r = torch.stack([r.relu for r in race])

        output,relu = model(Xt, p1=False)
        # print(output)
        relu = r.sum(dim=-1)

        _, actual = torch.max(y.data, 1)
        onehot_win = F.one_hot(actual, num_classes=8)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == actual).sum().item()

        #One hot wins
        label = torch.zeros_like(y.data).scatter_(1, torch.argmax(y.data, dim=1).unsqueeze(1), 1.)
        pred_label = torch.zeros_like(output.data).scatter_(1, torch.argmax(output.data, dim=1).unsqueeze(1), 1.)
        correct_tensor = label*pred_label

        correct_l = predicted == actual

        softmax_preds = sft_max(output)
        softmax_preds1 = sft_max(output/0.5)
        softmax_preds2 = sft_max(output/0.5)

        loss = criterion(output, y).mean()

        loss_tensor = validation_CLE(output,y)

        test_races = race

        price_tensor = torch.tensor([x.prices for x in test_races], device=device)

        profit_tensor = price_tensor*correct_tensor*0.95-pred_label

        # relu = torch.mean(abs(relu),dim=1)

        # print(relu)

        races = {}
        # races['race_ids'] = [x.raceid for x in test_races]

        # for r in race[race_ids]:
        #     if len(r)>

        races['raw_margins'] = [x.raw_margins for x in test_races]
        races['correct'] = correct_tensor.tolist()
        races['simple']  = profit_tensor.tolist()
        # races['output'] = [x.margins for x in test_races]
        races['relu'] = [[x]*8 for x in relu.cpu().tolist()]
        # print(races['relu'])
        races['pred_prob'] = softmax_preds.tolist()
        races['pred_prob2'] = softmax_preds2.tolist()
        races['prices'] = [x.prices for x in test_races]
        races['imp_prob'] = [x.implied_prob  for x in test_races]
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
        races['race_num'] = [[int(x.race_num)]*8 for x in race]
        races['loss'] = loss_tensor.tolist()


        accuracy = correct/len(predicted)


        races['one_hot_win'] = onehot_win.tolist()
        races['track'] = [[x.track_name]*8 for x in test_races]

        for k,value in races.items():
            # print(k)
            races[k] = [item for sublist in value for item in sublist]
            # print(len(races[k]))
        all_price_df = pd.DataFrame(races)

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
                    'val_multibet profit < 30':all_price_df['profit < 30'].sum(),
                    'val_multibet profit < 30 sd':all_price_df['profit < 30'].std(),
                    'val_multibet outlay < 30':all_price_df['outlay < 30'].sum(),
                    'val_ROI < 30':all_price_df['profit < 30'].sum()/all_price_df['outlay < 30'].sum(),
                    'val_ROI < 30 2':all_price_df['profit < 30 2'].sum()/all_price_df['outlay < 30 2'].sum(),
                    "val_loss_val":torch.mean(loss).item(),
                    'val_flat_simple':all_price_df['simple'].sum(),
                    'val_profit_relu':all_price_df['profit_relu'].sum(),
                    'val_relu roi':all_price_df['profit_relu'].sum()/all_price_df['bet_relu'].sum(),
                    'epoch':epoch,
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

        if epoch%100==0:
            all_price_df.reset_index().to_feather(f'./model_all_price/{wandb.run.name} - all_price_df.fth')
        
        wandb.log(stats_dict)
        wandb.log({"accuracy2": correct/len_test})

        return accuracy


def train_OHE_v3(model:GRUNetv3,raceDB:Races, criterion, optimizer,scheduler, config=None):
    # torch.autograd.set_detect_anomaly(True)

    last = 0
    if config!=None:
        batch_size = config['batch_size']
        epochs = config['epochs']
    else:
        batch_size = 25
    len_train_dogs = len(raceDB.train_dog_ids)
    len_train_races = len(raceDB.train_race_ids)
    example_ct = 0  # number of examples seen
    batch_ct = 0
    m = nn.LogSoftmax(dim=1)

    num_batches = raceDB.batches['num_batches']

    for epoch in trange(epochs):
        model.train()

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

                for i,dog in enumerate(train_dog_input):
                    [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[i])]

                [setattr(obj, 'hidden', val) for obj, val in zip(dogs,hidden)]


                # #NEEDED???
                # raceDB.margin_from_dog_to_race_v3(mode='train', batch_races=batch_races_ids)

                [setattr(race, 'hidden_in', torch.cat([race.race_dist]+[race.race_track]+[d.hidden_out for d in race.dogs])) for race in batch_races]

                race = batch_races

                X2 = torch.stack([r.hidden_in for r in race]) #Input for FFNN
                y = torch.stack([x.one_hot_class for x in race])
                w = torch.stack([x.new_win_weight for x in race])

                output = model(X2, p1=False)
                model.zero_grad(set_to_none=True)

                epoch_loss = criterion(output, y)*w

            epoch_loss.mean().backward()
            optimizer.step()
            
            wandb.log({"loss_1": torch.mean(epoch_loss).item()}, step = example_ct)
            raceDB.detach_hidden(dogs)

            for i,dog in enumerate(train_dog_input):
                [setattr(obj, 'hidden_out', val.detach()) for obj, val in zip(dog,output[i])]


        if (epoch)%20==0:
            validate_model_v3_OHE(model,raceDB, criterion=criterion, epoch=epoch)
        if (epoch+1)%100==0:
            raceDB.create_hidden_states_dict_v2()
            model_saver_wandb(model, optimizer, epoch, 0.1, raceDB.hidden_states_dict_gru_v6, raceDB.train_hidden_dict, model_name="long nsw new  22000 RUN")
        # elif (epoch)%3==0:
        #     validate_model_v3(model,raceDB, criterion=criterion, epoch=epoch) 




        raceDB.reset_hidden()

    return model

#Testing
@torch.no_grad()
def validate_model_v3_OHE(model:GRUNetv3,raceDB:Races,criterion=None, batch_size=None,epoch=10,config=None):
    # torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)

    correct = 0
    total = 0
    model.eval()

    price_dict = {}
    price_dict['prices'] = []
    price_dict['imp_prob'] = []
    price_dict['pred_prob'] = []
    price_dict['pred_price'] = []
    price_dict['margin'] = []
    price_dict['onehot_win'] = []
    price_dict['raceID'] = []
    price_dict['dogID'] = []
    price_dict['track'] = []
    price_dict['date'] = []
    price_dict['grade'] = []
    price_dict['loss'] = []
    race_ids = []
    # criterion = nn.CrossEntropyLoss(reduction='none')


    model.eval()


    with torch.no_grad():

        len_test = len(raceDB.val_dog_ids)
        val_idx = range(0,len_test)

        # dog_inputs = [[z.full_input for z in inner] for inner in [x for x in raceDB.train_dogs.values()]]
        # Uses train dogs here because only dogs in Train set will have new hidden values?? Does that make sense?
        # no will jumble around but wont throw error in feeding model, because when building dogs sorted, 
        # lengths will be made to match size of X(test)
        # Shouldn't have much of an effect due to size of test set??
        dogs = [x for x in  raceDB.test_dogs.values()]  #[Dog]
        dog_input = [inner for inner in [x for x in raceDB.get_dog_val(val_idx)]] #[[DogInput]]
    
        X = raceDB.batches['packed_y']

        # train = [torch.stack(n,0) for n in [[z.stats for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]]
        # X = pack_sequence(train, enforce_sorted=False).to('cuda:0')

        # dogs_sorted = [dogs[x] for x in X[3]]
        hidden_in = torch.stack([x.hidden for x in dogs]).transpose(0,1)

        # margins = [torch.stack(n,0) for n in[[torch.tensor(z.margin).to('cuda:0') for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]]
        # Y = pack_sequence(margins, enforce_sorted=False).data.view(-1,1)


        output,hidden = model(X,h=hidden_in) # Shape List[Tensor[Dog]]
        hidden = hidden.transpose(0,1)
        for i,dog in enumerate(dogs):
            dog.hidden_test = hidden[i]


        for i,dog in enumerate(dog_input):
                [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[i])]

        # for i,dog in enumerate(dog_input):
        #     dog_outputs = output[i]
        #     # print(f'{i=}, shape of output {len(dog_outputs)} type of output {type(dog_outputs)}')
        #     for j,di in enumerate(dog):
        #         di.hidden_out = dog_outputs[j]

        raceDB.margin_from_dog_to_race_v3(mode='test')

        len_test = len(raceDB.test_race_ids)
        test_idx = range(0,len_test)

        race = raceDB.get_test_input(test_idx)

        Xt = torch.stack([r.hidden_in for r in race]) #Input for FFNN
        y = torch.stack([x.one_hot_class for x in race])

        output = model(Xt, p1=False)
        # print(output)

        _, actual = torch.max(y.data, 1)
        onehot_win = F.one_hot(actual, num_classes=8)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == actual).sum().item()
        correct_l = predicted == actual

        softmax_preds = sft_max(output)
        softmax_preds1 = sft_max(output/1.5)
        softmax_preds2 = sft_max(output/1.7)

        loss = criterion(output, y).mean()



        test_races = raceDB.get_test_input(test_idx)

        races = {}
        races['race_ids'] = [x.raceid for x in test_races]
        races['raw_margins'] = [x.raw_margins for x in test_races]
        # races['output'] = [x.margins for x in test_races]
        races['pred_prob'] = softmax_preds.tolist()
        races['pred_prob2'] = softmax_preds2.tolist()
        races['prices'] = [x.prices for x in test_races]
        races['imp_prob'] = [x.implied_prob  for x in test_races]
        races['pred_price'] = (1/softmax_preds).tolist()
        races['pred_price1'] = (1/softmax_preds1).tolist()
        races['pred_price2'] = (1/softmax_preds2).tolist()
        # races['pred_prob'] = [x.tolist() for x in races['pred_prob']]
        races['classes'] = [x.classes.tolist() for x in test_races]
        races['track'] = [x.track_name for x in test_races]
        races['one_hot_win'] = [x.one_hot_class.tolist() for x in test_races]
        # races['date'] = [x.race_date for x in test_races]
        races['dogID'] = [x.list_dog_ids() for x in race]
        races['raceID'] = [[x.raceid]*8 for x in race]
        races['date'] = [[x.race_date]*8 for x in race]


        accuracy = correct/len(predicted)

        # for k,v in races.items():
        #     print(f"{k} length is {len(v)} type {type(v[0])}")

        


        races['one_hot_win'] = onehot_win.tolist()
        races['track'] = [[x.track_name]*8 for x in test_races]

        prices_flat = [item for sublist in races['prices'] for item in sublist]
        pred_prices = [item for sublist in races['pred_price'] for item in sublist]
        pred_prices1 = [item for sublist in races['pred_price1'] for item in sublist]
        pred_prices2 = [item for sublist in races['pred_price2'] for item in sublist]
        onehot_win  = [item for sublist in races['one_hot_win'] for item in sublist]
        flat_margins = [item for sublist in races['raw_margins'] for item in sublist]
        flat_track = [item for sublist in races['track'] for item in sublist]
        flat_dogs  = [item for sublist in races['dogID'] for item in sublist]
        flat_races = [item for sublist in races['raceID'] for item in sublist]
        flat_date  = [item for sublist in races['date'] for item in sublist]
        #flat_dogs  = [item for sublist in races['dogID'].tolist() for item in sublist]
        #flat_races = [item for sublist in prices_df['raceID'].tolist() for item in sublist]
        #flat_track = [item for sublist in prices_df['track'].tolist() for item in sublist]
        #flat_date  = [item for sublist in prices_df['date'].tolist() for item in sublist]
        #flat_grade  = [item for sublist in prices_df['grade'].tolist() for item in sublist]
        #flat_loss  = [item for sublist in prices_df['loss'].tolist() for item in sublist]

        all_price_df = pd.DataFrame(data={'flat_dogs':flat_dogs,'flat_races':flat_races,'flat_date':flat_date,'track':flat_track, 'prices':prices_flat,'pred_prices2':pred_prices2, 'pred_price':pred_prices, 'onehot_win':onehot_win,'split_margin':flat_margins})
        all_price_df = all_price_df[all_price_df['prices']>1]
        all_price_df['imp_prob'] =  all_price_df.apply(lambda x: 1/x['prices'], axis = 1)
        all_price_df['pred_price'] =  all_price_df['pred_price'].clip(0,100)
        all_price_df['pred_prob'] =  all_price_df.apply(lambda x: 1/x['pred_price'], axis = 1)
        all_price_df['pred_prob2'] =  all_price_df.apply(lambda x: 1/x['pred_prices2'], axis = 1)
        all_price_df['bet amount'] = all_price_df.apply(lambda x: (x['pred_prob']-x['imp_prob'])*100 if (x['pred_prob']>x['imp_prob'])&(1>x['imp_prob']>0) else 0, axis = 1)
        all_price_df['profit'] = all_price_df.apply(lambda x: x['bet amount']*(x['prices']-1)*0.95 if x['onehot_win'] else -1*x['bet amount'], axis = 1)
        all_price_df['win_price'] = all_price_df.apply(lambda x: x['prices'] if (x['onehot_win']) else 0, axis = 1)
        all_price_df['colour'] = all_price_df.apply(lambda x: "profitz" if x['profit']>0 else ("loss" if x['profit']<0 else ("no bet - win" if x['onehot_win'] else "no bet")), axis=1)
        all_price_df['profit < 30'] = all_price_df.apply(lambda x: x['profit'] if x['prices']<30 else 0, axis=1)
        all_price_df['outlay < 30'] = all_price_df.apply(lambda x: x['bet amount'] if x['prices']<30 else 0, axis=1)

        all_price_df['bet amount2'] = all_price_df.apply(lambda x: (x['pred_prob2']-x['imp_prob'])*100 if (x['pred_prob2']>x['imp_prob'])&(1>x['imp_prob']>0) else 0, axis = 1)
        all_price_df['profit2'] = all_price_df.apply(lambda x: x['bet amount2']*(x['prices']-1)*0.95 if x['onehot_win'] else -1*x['bet amount2'], axis = 1)
        all_price_df['profit < 30 2'] = all_price_df.apply(lambda x: x['profit2'] if x['prices']<30 else 0, axis=1)
        all_price_df['outlay < 30 2'] = all_price_df.apply(lambda x: x['bet amount2'] if x['prices']<30 else 0, axis=1)



        flat_track_df = all_price_df.groupby('track').sum(numeric_only=True).reset_index()
        flat_track_df['ROI < 30'] = flat_track_df.apply(lambda x: x['profit < 30']/x['outlay < 30'],axis =1)
        flat_track_df['ROI < 30 2'] = flat_track_df.apply(lambda x: x['profit < 30 2']/x['outlay < 30 2'],axis =1)

        all_price_df.to_csv(f'./model_all_price/{wandb.run.name} - all_price_df.csv')

        if epoch%10==0:
            try:
                wandb.log({"all_price_df":wandb.Table(dataframe=pd.DataFrame(all_price_df))})
                basic = wandb.log({"basic_table":wandb.Table(dataframe=pd.DataFrame(races))})
                wandb.log({'flat_track':wandb.Table(dataframe=flat_track_df)})
            except Exception as e:
                pass
        else:
            try:
                wandb.log({'flat_track':wandb.Table(dataframe=flat_track_df)})
            except Exception as e:
                pass

        

        wandb.log({"accuracy": correct/len_test,
                    'multibet profit':all_price_df['profit'].sum(),
                    'multibet profit < 30':all_price_df['profit < 30'].sum(),
                    'multibet profit < 30 sd':all_price_df['profit < 30'].std(),
                    'multibet outlay < 30':all_price_df['outlay < 30'].sum(),
                    'ROI < 30':all_price_df['profit < 30'].sum()/all_price_df['outlay < 30'].sum(),
                    'ROI < 30 2':all_price_df['profit < 30 2'].sum()/all_price_df['outlay < 30 2'].sum(),
                    "loss_val":torch.mean(loss)})

        return accuracy