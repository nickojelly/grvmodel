from rnn_classes import *
import pickle
import pandas as pd
import wandb
import os

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



#Testing
def validate_model(model:GRUNet,raceDB:Races,criterion, batch_size, example_ct, epoch_loss, batch_ct,epoch,config):
    torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)
    l_sftmax = nn.LogSoftmax(dim=-1)

    batch_size= 2000
    len_test = len(raceDB.test_race_ids)
    num_batches = len_test/batch_size
    list_t = [] 
    last = 0
    loss_val = 0 
    correct = 0
    total = 0
    model.eval()
    actuals = []
    preds = []
    grades = []
    tracks = []
    pred_confs = []
    bfsps = []
    start_prices = []
    loss_l = []
    loss_t = []
    margins_l = []
    preds_l = []
    pred_sftmax = []
    raw_margins = []
    raw_places = []
    margins_prob = []

    prices_list = []
    raw_prices = []

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

    model.eval()

    loss_dict = {}

    with torch.no_grad():
        races_idx = range(0,len(raceDB.test_race_ids)-1)
        race = raceDB.get_test_input(races_idx)

        X = race
        y = torch.stack([x.classes for x in race])
        output = model(X)
        race_ids = [x.raceid for x in race]

        _, actual = torch.max(y.data, 1)
        onehot_win = F.one_hot(actual, num_classes=8)
        conf, predicted = torch.max(output.data, 1)
        correct += (predicted == actual).sum().item()

        softmax_preds = sft_max(output)

        
        total += len(raceDB.test_race_ids)
        actuals = actual.tolist()
        preds = predicted.tolist()
        pred_confs = conf.tolist()
        tracks = [r.track_name for r in race]
        grades = [r.grade for r in race]
        for i,dog_idx in enumerate(actual.tolist()):
            bfsps.append(race[i].dogs[dog_idx].bfsp)
            #start_prices.append(race[i].dogs[dog_idx].sp)

        
        loss = criterion(output, y).detach()
        loss_tensor = validation_CLE(output,y)
        loss_dict['loss_t'] = loss_tensor.tolist()

        loss_dict['loss_l'] = loss.tolist()
        loss_dict['preds_l'] = output.tolist()
        loss_dict['pred_sftmax'] = softmax_preds.tolist()
        loss_dict['margins_l'] = y.tolist()
        loss_dict['margins_prob'] = y.tolist()
        loss_dict['raw_margins'] = [x.raw_margins for x in race]
        loss_dict['raw_places'] = [x.raw_places for x in race]
        loss_val += loss.mean()

        price_dict['prices'] = [x.prices for x in race]
        price_dict['loss'] = loss_tensor.tolist()
        price_dict['imp_prob'] = [x.implied_prob for x in race]
        price_dict['pred_prob'] = softmax_preds.tolist()
        #print([(1/(x+(-7**10))).tolist() for x in torch.exp(output)])
        price_dict['pred_price'] = [(1/(x)).tolist() for x in softmax_preds]
        price_dict['margin'] = [x.raw_margins for x in race]
        price_dict['onehot_win'] = onehot_win.tolist()
        price_dict['raceID'] = [[x.raceid]*8 for x in race]
        # price_dict['raceID'] = [[x.raceid]*8 for x in race])
        price_dict['dogID'] = [x.list_dog_ids() for x in race]
        price_dict['track'] = [[x.track_name]*8 for x in race]
        price_dict['date'] = [[x.race_date]*8 for x in race]
        price_dict['grade'] = [[x.grade]*8 for x in race]


        races_idx = range(0,len(raceDB.test_race_ids))
        race = raceDB.get_test_input(races_idx)


        loss_list = []

        #print("start loss calc")
        # for i,l in enumerate(loss_l):
        #     for j in range(0,7):
        #         loss_list.append([preds_l[i][j],margins_l[i][j],loss_t[i][j],l[j],pred_sftmax[i][j],margins_prob[i][j], raw_margins[i][j], raw_places[i][j]])

    # for k,v in loss_dict.items():
    #     print(f'length of {k} = {len(v)}')

    loss_df = pd.DataFrame(loss_dict)
    loss_table = wandb.Table(dataframe=loss_df)


    logdf = pd.DataFrame(data = {"actuals":actuals, "preds":preds,"conf":pred_confs, "grade":grades, "track":tracks, "bfsps":bfsps})#, "sp":start_prices })
    
    logdf['correct'] = logdf.apply(lambda x: 1 if x['actuals']==x['preds'] else 0, axis=1)
    logdf['profit'] = logdf.apply(lambda x: 0 if x['bfsps']<1 else x['bfsps']-1  if x['correct'] else -1, axis=1)
    logdf.to_csv('logDFtest.csv')

    table = wandb.Table(dataframe=logdf)

    logdf['count'] = 1
    logdf['eligible_ct'] = logdf.apply(lambda x: 1 if x['bfsps'] > 1 else 0, axis = 1)

    track_df = logdf.groupby('track', as_index=False).sum(numeric_only=True).reset_index()
    track_table = wandb.Table(dataframe=track_df)
    
    prices_df = pd.DataFrame(price_dict)


    prices_df['sum_price'] = prices_df.apply(lambda x: sum(x['prices']), axis = 1)
    prices_df = prices_df[prices_df['sum_price']>0]

    prices_flat = [item for sublist in prices_df['prices'].tolist() for item in sublist]
    pred_prices = [item for sublist in prices_df['pred_price'].tolist() for item in sublist]
    onehot_win  = [item for sublist in prices_df['onehot_win'].tolist() for item in sublist]
    flat_margins = [item for sublist in prices_df['margin'].tolist() for item in sublist]
    flat_dogs  = [item for sublist in prices_df['dogID'].tolist() for item in sublist]
    flat_races = [item for sublist in prices_df['raceID'].tolist() for item in sublist]
    flat_track = [item for sublist in prices_df['track'].tolist() for item in sublist]
    flat_date  = [item for sublist in prices_df['date'].tolist() for item in sublist]
    flat_grade  = [item for sublist in prices_df['grade'].tolist() for item in sublist]
    flat_loss  = [item for sublist in prices_df['loss'].tolist() for item in sublist]
    all_price_df = pd.DataFrame(data={'flat_dogs':flat_dogs,'flat_races':flat_races,'prices':prices_flat, 'pred_price':pred_prices, 'onehot_win':onehot_win,'split_margin':flat_margins,'flat_track':flat_track,'flat_date':flat_date,'flat_grade':flat_grade,'flat_loss':flat_loss})
    all_price_df = all_price_df[all_price_df['prices']>1]
    all_price_df['imp_prob'] =  all_price_df.apply(lambda x: 1/x['prices'], axis = 1)
    all_price_df['pred_price'] =  all_price_df['pred_price'].clip(0,100)
    all_price_df['pred_prob'] =  all_price_df.apply(lambda x: 1/x['pred_price'], axis = 1)
    all_price_df['bet amount'] = all_price_df.apply(lambda x: (x['pred_prob']-x['imp_prob'])*100 if (x['pred_prob']>x['imp_prob'])&(1>x['imp_prob']>0) else 0, axis = 1)
    all_price_df['frac kel'] = all_price_df.apply(lambda x: (x['pred_prob']-(1-x['pred_prob'])/x['prices'])*100 if (x['pred_prob']>x['imp_prob'])&(1>x['imp_prob']>0) else 0, axis = 1)
    all_price_df['frac profit'] = all_price_df.apply(lambda x: x['frac kel']*(x['prices']-1)*0.95 if x['onehot_win'] else -1*x['frac kel'], axis = 1)
    all_price_df['profit'] = all_price_df.apply(lambda x: x['bet amount']*(x['prices']-1)*0.95 if x['onehot_win'] else -1*x['bet amount'], axis = 1)
    all_price_df['flat_profit'] = all_price_df.apply(lambda x: x['prices'] if (x['onehot_win'] and x['bet amount']) else 0, axis = 1)
    all_price_df['win_price'] = all_price_df.apply(lambda x: x['prices'] if (x['onehot_win']) else 0, axis = 1)
    all_price_df['colour'] = all_price_df.apply(lambda x: "profitz" if x['profit']>0 else ("loss" if x['profit']<0 else ("no bet - win" if x['onehot_win'] else "no bet")), axis=1)
    all_price_df['profit < 30'] = all_price_df.apply(lambda x: x['profit'] if x['prices']<30 else 0, axis=1)
    all_price_df['outlay < 30'] = all_price_df.apply(lambda x: x['bet amount'] if x['prices']<30 else 0, axis=1)
    all_price_df['fk profit < 30'] = all_price_df.apply(lambda x: x['frac profit'] if x['prices']<30 else 0, axis=1)
    all_price_df['fk outlay < 30'] = all_price_df.apply(lambda x: x['frac kel'] if x['prices']<30 else 0, axis=1)
    #all_price_df['correct'] = all_price_df.apply(lambda x: x['frac kel'] if x['prices']<30 else 0, axis=1)

    all_price_table = wandb.Table(dataframe=all_price_df)
    flat_track_df = all_price_df.groupby('flat_track').sum(numeric_only=True).reset_index()
    flat_track_df['ROI < 30'] = flat_track_df.apply(lambda x: x['profit < 30']/x['outlay < 30'],axis =1)
    flat_track_df['loss avg'] = flat_track_df.apply(lambda x: x['flat_loss']/x['onehot_win'], axis=1)
    flat_track_df['profit_captured'] = flat_track_df.apply(lambda x: x['flat_profit']/x['win_price'], axis=1)
    all_price_df.to_pickle('all_price_df_nz.npy')
    try:
        wandb.log({'flat_track':wandb.Table(dataframe=flat_track_df)})
    except Exception as e:
        pass
    # all_price_df.to_excel(f'./model_all_price/{wandb.run.name} - all_price_df.xlsx')


    price_table = wandb.Table(dataframe=prices_df)

    if epoch%20+1==0:
        try:
            wandb.log({"table_key": table,"loss_table": loss_table,"track_df": track_table,"price_table":price_table, 'allprice_df':all_price_df})
        except Exception as e:
            print(e)

    #print(f"accuray: {correct/total}")
    wandb.log({"accuracy": correct/total, 
                "loss_val": torch.mean(loss_val)/num_batches, 
                "correct": correct, 
                'profit': logdf[logdf['actuals']==logdf['preds']]['profit'].sum(), 
                'multibet profit':all_price_df['profit'].sum(),
                'multibet profit < 30':all_price_df['profit < 30'].sum(),
                'multibet profit < 30 sd':all_price_df['profit < 30'].std(),
                'multibet outlay < 30':all_price_df['outlay < 30'].sum(),
                'ROI < 30':all_price_df['profit < 30'].sum()/all_price_df['outlay < 30'].sum(),
                'multibet profit sd':all_price_df['profit'].std(), 
                'multibet outlay':all_price_df['bet amount'].sum(),
                'ROI': all_price_df['profit'].sum()/all_price_df['bet amount'].sum(),
                'FK ROI < 30': all_price_df['fk profit < 30'].sum()/all_price_df['fk outlay < 30'].sum(),
                'FK ROI': all_price_df['frac profit'].sum()/all_price_df['frac kel'].sum()})

    return correct/total, torch.mean(loss_val)

#Testing
def validate_model_L1(model:GRUNet,raceDB:Races,criterion, batch_size, example_ct, epoch_loss, batch_ct,epoch,config):
    torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)
    sft_min = nn.Softmin(dim=1)
    l_sftmax = nn.LogSoftmax(dim=-1)

    batch_size= max([int(config['batch_size']/10),25])
    len_test = len(raceDB.test_race_ids)-batch_size
    num_batches = len_test/batch_size
    list_t = [] 
    last = 0
    loss_val = 0 
    correct = 0
    total = 0
    model.eval()
    actuals = []
    preds = []
    grades = []
    tracks = []
    pred_confs = []
    bfsps = []
    start_prices = []
    loss_l = []
    loss_t = []
    margins_l = []
    preds_l = []
    pred_sftmax = []
    raw_margins = []
    raw_places = []
    margins_prob = []

    prices_list = []
    raw_prices = []

    price_dict = {}
    price_dict['prices'] = []
    price_dict['imp_prob'] = []
    price_dict['pred_prob'] = []
    price_dict['pred_price'] = []
    price_dict['margin'] = []
    price_dict['onehot_win'] = []
    price_dict['raceID'] = []
    price_dict['dogID'] = []
    race_ids = []


    with torch.no_grad():
        # loss_comparison
        # full_test_races = raceDB.get_test_input(range(0,len(raceDB.test_race_ids)))
        # output = l_sftmax(model(full_test_races))
        # bf_prices = l_sftmax(torch.tensor([x.prices for x in full_test_races]).to('cuda:0'))
        # full_classes = torch.stack([x.classes for x in full_test_races])
        
        # print(f"our loss = {nnl_loss(output,full_classes)}")
        # print(f"their loss = {nnl_loss(output,full_classes)}")

        for i in trange(0,len_test,batch_size, leave=False):
            races_idx = range(last,last+batch_size)
            last = i+batch_size
            race = raceDB.get_test_input(races_idx)
            #tracks.extend([r.track for r in race])

            X = race
            y = torch.stack([x.classes for x in race])
            output = model(X)
            race_ids.extend([x.raceid for x in race])
            #print(y)

            _, actual = torch.max(y.data, 1)
            onehot_win = F.one_hot(actual, num_classes=8)
            conf, predicted = torch.max(output.data, 1)
            correct += (predicted == actual).sum().item()

            softmax_preds = sft_max(output)

            
            total += batch_size
            actuals.extend(actual.tolist())
            preds.extend(predicted.tolist())
            pred_confs.extend(conf.tolist())
            tracks.extend([r.track_name for r in race])
            grades.extend([r.grade for r in race])
            for i,dog_idx in enumerate(actual.tolist()):
                bfsps.append(race[i].dogs[dog_idx].bfsp)
                #start_prices.append(race[i].dogs[dog_idx].sp)

            
            loss = criterion(output, y).detach()
            loss_tensor = validation_CLE(output,y)
            loss_t.append(loss_tensor.tolist())

            loss_l.append(loss.tolist())
            preds_l.append(output.tolist())
            pred_sftmax.append(softmax_preds.tolist())
            margins_l.append(y.tolist())
            margins_prob.append(y.tolist())
            raw_margins.append([x.raw_margins for x in race])
            raw_places.append([x.raw_places for x in race])
            loss_val += loss

            price_dict['prices'].extend([x.prices for x in race])
            price_dict['imp_prob'].extend([x.implied_prob for x in race])
            price_dict['pred_prob'].extend(softmax_preds.tolist())
            #price_dict['pred_prob'].extend([x.tolist() for x in torch.exp(output)])
            #print([(1/(x+(-7**10))).tolist() for x in torch.exp(output)])
            price_dict['pred_price'].extend([(1/(x+10**-3)).tolist() for x in softmax_preds])
            price_dict['margin'].extend([x.raw_margins for x in race])
            price_dict['onehot_win'].extend(onehot_win.tolist())
            price_dict['raceID'].extend([[x.raceid]*8 for x in race])
            price_dict['dogID'].extend([x.list_dog_ids() for x in race])

            

        loss_list = []

        #print("start loss calc")
        for i,l in enumerate(loss_l):
            for j in range(0,7):
                loss_list.append([preds_l[i][j],margins_l[i][j],loss_t[i][j],l[j],pred_sftmax[i][j],margins_prob[i][j], raw_margins[i][j], raw_places[i][j]])
    loss_df = pd.DataFrame(loss_list, columns=['Preds','Margins','Indiviual Losses','Losses','softmaxPreds','Softmax Margins','Raw margins', 'Raw places'])
    loss_table = wandb.Table(dataframe=loss_df)


    logdf = pd.DataFrame(data = {"actuals":actuals, "preds":preds,"conf":pred_confs, "grade":grades, "track":tracks, "bfsps":bfsps})#, "sp":start_prices })
    
    logdf['correct'] = logdf.apply(lambda x: 1 if x['actuals']==x['preds'] else 0, axis=1)
    logdf['profit'] = logdf.apply(lambda x: 0 if x['bfsps']<1 else x['bfsps']-1  if x['correct'] else -1, axis=1)
    logdf.to_csv('logDFtest.csv')

    table = wandb.Table(dataframe=logdf)

    logdf['count'] = 1
    logdf['eligible_ct'] = logdf.apply(lambda x: 1 if x['bfsps'] > 1 else 0, axis = 1)

    track_df = logdf.groupby('track', as_index=False).sum(numeric_only=True).reset_index()
    track_table = wandb.Table(dataframe=track_df)
    
    prices_df = pd.DataFrame(price_dict)


    prices_df['sum_price'] = prices_df.apply(lambda x: sum(x['prices']), axis = 1)
    prices_df = prices_df[prices_df['sum_price']>0]

    prices_flat = [item for sublist in prices_df['prices'].tolist() for item in sublist]
    pred_prices = [item for sublist in prices_df['pred_price'].tolist() for item in sublist]
    onehot_win  = [item for sublist in prices_df['onehot_win'].tolist() for item in sublist]
    flat_margins = [item for sublist in prices_df['margin'].tolist() for item in sublist]
    flat_dogs  = [item for sublist in prices_df['dogID'].tolist() for item in sublist]
    flat_races = [item for sublist in prices_df['raceID'].tolist() for item in sublist]
    flat_track = [item for sublist in prices_df['track'].tolist() for item in sublist]
    flat_date  = [item for sublist in prices_df['date'].tolist() for item in sublist]
    all_price_df = pd.DataFrame(data={'flat_dogs':flat_dogs,'flat_races':flat_races,'prices':prices_flat, 'pred_price':pred_prices, 'onehot_win':onehot_win,'split_margin':flat_margins,'flat_track':flat_track,'flat_date':flat_date})
    all_price_df = all_price_df[all_price_df['prices']>1]
    all_price_df['imp_prob'] =  all_price_df.apply(lambda x: 1/x['prices'], axis = 1)
    all_price_df['pred_prob'] =  all_price_df.apply(lambda x: 1/x['pred_price'], axis = 1)
    all_price_df['bet amount'] = all_price_df.apply(lambda x: (x['pred_prob']-x['imp_prob'])*100 if (x['pred_prob']>x['imp_prob'])&(1>x['imp_prob']>0) else 0, axis = 1)
    all_price_df['frac kel'] = all_price_df.apply(lambda x: (x['pred_prob']-(1-x['pred_prob'])/x['prices'])*100 if (x['pred_prob']>x['imp_prob'])&(1>x['imp_prob']>0) else 0, axis = 1)
    all_price_df['frac profit'] = all_price_df.apply(lambda x: x['frac kel']*(x['prices']-1)*0.95 if x['onehot_win'] else -1*x['frac kel'], axis = 1)
    all_price_df['profit'] = all_price_df.apply(lambda x: x['bet amount']*(x['prices']-1)*0.95 if x['onehot_win'] else -1*x['bet amount'], axis = 1)
    all_price_df['flat_profit'] = all_price_df.apply(lambda x: 1*(x['prices']-1)*0.95 if (x['onehot_win'] and x['bet amount']) else -1, axis = 1)
    all_price_df['colour'] = all_price_df.apply(lambda x: "profitz" if x['profit']>0 else ("loss" if x['profit']<0 else ("no bet - win" if x['onehot_win'] else "no bet")), axis=1)
    all_price_df['profit < 30'] = all_price_df.apply(lambda x: x['profit'] if x['prices']<30 else 0, axis=1)
    all_price_df['outlay < 30'] = all_price_df.apply(lambda x: x['bet amount'] if x['prices']<30 else 0, axis=1)
    all_price_df['fk profit < 30'] = all_price_df.apply(lambda x: x['frac profit'] if x['prices']<30 else 0, axis=1)
    all_price_df['fk outlay < 30'] = all_price_df.apply(lambda x: x['frac kel'] if x['prices']<30 else 0, axis=1)
    #all_price_df['correct'] = all_price_df.apply(lambda x: x['frac kel'] if x['prices']<30 else 0, axis=1)

    all_price_table = wandb.Table(dataframe=all_price_df)
    flat_track_df = all_price_df.groupby('flat_track').sum(numeric_only=True).reset_index()
    flat_track_df['ROI < 30'] = flat_track_df.apply(lambda x: x['profit < 30']/x['outlay < 30'],axis =1)
    all_price_df.to_pickle('all_price_df_nz.npy')
    try:
        wandb.log({'flat_track':wandb.Table(dataframe=flat_track_df)})
    except Exception as e:
        pass
    all_price_df.to_excel(f'./model_all_price/{wandb.run.name} - all_price_df.xlsx')


    price_table = wandb.Table(dataframe=prices_df)

    if epoch%10==0:
        try:
            wandb.log({"table_key": table,"loss_table": loss_table,"track_df": track_table,"price_table":price_table, 'allprice_df':all_price_df})
        except Exception as e:
            print(e)

    #print(f"accuray: {correct/total}")
    wandb.log({"accuracy": correct/total, 
                "loss_val": torch.mean(loss_val)/num_batches, 
                "correct": correct, 
                'profit': logdf[logdf['actuals']==logdf['preds']]['profit'].sum(), 
                'multibet profit':all_price_df['profit'].sum(),
                'multibet profit < 30':all_price_df['profit < 30'].sum(),
                'multibet profit < 30 sd':all_price_df['profit < 30'].std(),
                'multibet outlay < 30':all_price_df['outlay < 30'].sum(),
                'ROI < 30':all_price_df['profit < 30'].sum()/all_price_df['outlay < 30'].sum(),
                'multibet profit sd':all_price_df['profit'].std(), 
                'multibet outlay':all_price_df['bet amount'].sum(),
                'ROI': all_price_df['profit'].sum()/all_price_df['bet amount'].sum(),
                'FK ROI < 30': all_price_df['fk profit < 30'].sum()/all_price_df['fk outlay < 30'].sum(),
                'FK ROI': all_price_df['frac profit'].sum()/all_price_df['frac kel'].sum()})

    return correct/total, torch.mean(loss_val)

#Testing
def validate_model_KL(model:GRUNet,raceDB:Races,criterion, batch_size, example_ct, epoch_loss, batch_ct):
    torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)
    l_sftmax = nn.LogSoftmax(dim=-1)

    batch_size=10
    len_test = len(raceDB.test_race_ids)-batch_size
    list_t = [] 
    last = 0
    loss_val = 0 
    correct = 0
    total = 0
    model.eval()
    actuals = []
    preds = []
    grades = []
    tracks = []
    pred_confs = []
    bfsps = []
    start_prices = []
    loss_l = []
    loss_t = []
    margins_l = []
    preds_l = []
    pred_sftmax = []
    raw_margins = []
    raw_places = []
    margins_prob = []

    prices_list = []
    raw_prices = []

    price_dict = {}
    price_dict['prices'] = []
    price_dict['imp_prob'] = []
    price_dict['pred_prob'] = []
    price_dict['pred_price'] = []
    price_dict['margin'] = []
    price_dict['onehot_win'] = []
    price_dict['raceID'] = []
    price_dict['dogID'] = []
    race_ids = []


    with torch.no_grad():
        # loss_comparison
        # full_test_races = raceDB.get_test_input(range(0,len(raceDB.test_race_ids)))
        # output = l_sftmax(model(full_test_races))
        # bf_prices = l_sftmax(torch.tensor([x.prices for x in full_test_races]).to('cuda:0'))
        # full_classes = torch.stack([x.classes for x in full_test_races])
        
        # print(f"our loss = {nnl_loss(output,full_classes)}")
        # print(f"their loss = {nnl_loss(output,full_classes)}")

        for i in trange(0,len_test,batch_size, leave=False):
            races_idx = range(last,last+batch_size)
            last = i+batch_size
            race = raceDB.get_test_input(races_idx)
            #tracks.extend([r.track for r in race])

            X = race
            y = torch.stack([x.classes for x in race])
            output = model(X)
            race_ids.extend([x.raceid for x in race])
            #print(y)

            _, actual = torch.max(y.data, 1)
            onehot_win = F.one_hot(actual, num_classes=8)
            conf, predicted = torch.max(output.data, 1)
            correct += (predicted == actual).sum().item()

            softmax_preds = torch.exp(output)

            
            total += batch_size
            actuals.extend(actual.tolist())
            preds.extend(predicted.tolist())
            pred_confs.extend(conf.tolist())
            tracks.extend([r.track_name for r in race])
            grades.extend([r.grade for r in race])
            for i,dog_idx in enumerate(actual.tolist()):
                bfsps.append(race[i].dogs[dog_idx].bfsp)
                #start_prices.append(race[i].dogs[dog_idx].sp)

            
            loss = criterion(output, y).detach()
            loss_tensor = validation_CLE(output,y)
            loss_t.append(loss_tensor.tolist())

            loss_l.append(loss.tolist())
            preds_l.append(output.tolist())
            pred_sftmax.append(softmax_preds.tolist())
            margins_l.append(y.tolist())
            margins_prob.append(y.tolist())
            raw_margins.append([x.raw_margins for x in race])
            raw_places.append([x.raw_places for x in race])
            loss_val += loss

            price_dict['prices'].extend([x.prices for x in race])
            price_dict['imp_prob'].extend([x.implied_prob for x in race])
            price_dict['pred_prob'].extend(softmax_preds.tolist())
            #price_dict['pred_prob'].extend([x.tolist() for x in torch.exp(output)])
            #print([(1/(x+(-7**10))).tolist() for x in torch.exp(output)])
            price_dict['pred_price'].extend([(1/(x)).tolist() for x in softmax_preds])
            price_dict['margin'].extend([x.raw_margins for x in race])
            price_dict['onehot_win'].extend(onehot_win.tolist())
            price_dict['raceID'].extend([[x.raceid]*8 for x in race])
            price_dict['dogID'].extend([x.list_dog_ids() for x in race])

            

        loss_list = []

        #print("start loss calc")
        for i,l in enumerate(loss_l):
            for j in range(0,7):
                loss_list.append([preds_l[i][j],margins_l[i][j],loss_t[i][j],l[j],pred_sftmax[i][j],margins_prob[i][j], raw_margins[i][j], raw_places[i][j]])

    loss_df = pd.DataFrame(loss_list, columns=['Preds','Margins','Indiviual Losses','Losses','softmaxPreds','Softmax Margins','Raw margins', 'Raw places'])
    loss_table = wandb.Table(dataframe=loss_df)


    logdf = pd.DataFrame(data = {"actuals":actuals, "preds":preds,"conf":pred_confs, "grade":grades, "track":tracks, "bfsps":bfsps})#, "sp":start_prices })
    
    logdf['correct'] = logdf.apply(lambda x: 1 if x['actuals']==x['preds'] else 0, axis=1)
    logdf['profit'] = logdf.apply(lambda x: 0 if x['bfsps']<1 else x['bfsps']-1  if x['correct'] else -1, axis=1)
    logdf.to_csv('logDFtest.csv')

    table = wandb.Table(dataframe=logdf)

    logdf['count'] = 1
    logdf['eligible_ct'] = logdf.apply(lambda x: 1 if x['bfsps'] > 1 else 0, axis = 1)

    track_df = logdf.groupby('track', as_index=False).sum(numeric_only=True).reset_index()
    track_table = wandb.Table(dataframe=track_df)
    
    prices_df = pd.DataFrame(price_dict)


    prices_df['sum_price'] = prices_df.apply(lambda x: sum(x['prices']), axis = 1)
    prices_df = prices_df[prices_df['sum_price']>0]

    prices_flat = [item for sublist in prices_df['prices'].tolist() for item in sublist]
    pred_prices = [item for sublist in prices_df['pred_price'].tolist() for item in sublist]
    onehot_win  = [item for sublist in prices_df['onehot_win'].tolist() for item in sublist]
    flat_margins = [item for sublist in prices_df['margin'].tolist() for item in sublist]
    flat_dogs = [item for sublist in prices_df['dogID'].tolist() for item in sublist]
    flat_races = [item for sublist in prices_df['raceID'].tolist() for item in sublist]
    all_price_df = pd.DataFrame(data={'flat_dogs':flat_dogs,'flat_races':flat_races,'prices':prices_flat, 'pred_price':pred_prices, 'onehot_win':onehot_win,'split_margin':flat_margins})
    all_price_df = all_price_df[all_price_df['prices']>1]
    all_price_df['imp_prob'] =  all_price_df.apply(lambda x: 1/x['prices'], axis = 1)
    all_price_df['pred_prob'] =  all_price_df.apply(lambda x: 1/x['pred_price'], axis = 1)
    all_price_df['bet amount'] = all_price_df.apply(lambda x: (x['pred_prob']-x['imp_prob'])*100 if (x['pred_prob']>x['imp_prob'])&(1>x['imp_prob']>0) else 0, axis = 1)
    all_price_df['profit'] = all_price_df.apply(lambda x: x['bet amount']*(x['prices']-1)*0.95 if x['onehot_win'] else -1*x['bet amount'], axis = 1)
    all_price_df['flat_profit'] = all_price_df.apply(lambda x: 1*(x['prices']-1)*0.95 if (x['onehot_win'] and x['bet amount']) else -1, axis = 1)
    all_price_df['colour'] = all_price_df.apply(lambda x: "profitz" if x['profit']>0 else ("loss" if x['profit']<0 else ("no bet - win" if x['onehot_win'] else "no bet")), axis=1)

    

    all_price_table = wandb.Table(dataframe=all_price_df)

    all_price_df.to_pickle('all_price_df_nz.npy')


    price_table = wandb.Table(dataframe=prices_df)

    try:
        wandb.log({"table_key": table,"loss_table": loss_table,"track_df": track_table,"price_table":price_table, 'allprice_df':all_price_df })
    except Exception as e:
        print(e)

    #print(f"accuray: {correct/total}")
    wandb.log({"accuracy": correct/total, 
                "loss_val": torch.mean(loss_val)/len_test, 
                "correct": correct, 
                'profit': logdf[logdf['actuals']==logdf['preds']]['profit'].sum(), 
                'multibet profit':all_price_df['profit'].sum(), 
                'multibet outlay':all_price_df['bet amount'].sum() })

    return correct/total, torch.mean(loss_val)
#Testing
def validate_modelv2(model:GRUNet,raceDB:Races):
    torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)
    batch_size=10
    len_test = len(raceDB.raceIDs)-batch_size
    list_t = [] 
    last = 0
    loss_val = 0 
    correct = 0
    total = 0
    model.eval()
    actuals = []
    preds = []
    grades = []
    tracks = []
    pred_confs = []
    bfsps = []
    start_prices = []
    loss_l = []
    loss_t = []
    margins_l = []
    preds_l = []
    pred_sftmax = []
    raw_margins = []
    raw_places = []
    margins_prob = []

    prices_list = []
    raw_prices = []

    price_dict = {}
    price_dict['prices'] = []
    price_dict['imp_prob'] = []
    price_dict['pred_prob'] = []
    price_dict['pred_price'] = []
    price_dict['margin'] = []
    price_dict['onehot_win'] = []
    with torch.no_grad():
        for i in trange(0,len_test,batch_size, leave=False):
            races_idx = range(last,last+batch_size)
            last = i
            race = raceDB.get_race_input(races_idx)
            #tracks.extend([r.track for r in race])
            X = race
            y = torch.stack([x.classes for x in race])
            output = model(X)
            #print(y)

            _, actual = torch.max(y.data, 1)
            onehot_win = F.one_hot(actual, num_classes=8)
            conf, predicted = torch.max(output.data, 1)
            correct += (predicted == actual).sum().item()

            softmax_preds = sft_max(output)

            
            total += batch_size
            actuals.extend(actual.tolist())
            preds.extend(predicted.tolist())
            pred_confs.extend(conf.tolist())
            tracks.extend([r.track_name for r in race])
            grades.extend([r.grade for r in race])
            for i,dog_idx in enumerate(actual.tolist()):
                bfsps.append(race[i].dogs[dog_idx].bfsp)
                #start_prices.append(race[i].dogs[dog_idx].sp)


            price_dict['prices'].extend([x.prices for x in race])
            price_dict['imp_prob'].extend([x.implied_prob for x in race])
            price_dict['pred_prob'].extend(softmax_preds.tolist())
            #print([(1/(x+(-7**10))).tolist() for x in torch.exp(output)])
            price_dict['pred_price'].extend([(1/(x)).tolist() for x in softmax_preds])
            price_dict['margin'].extend([x.raw_margins for x in race])
            price_dict['onehot_win'].extend(onehot_win.tolist())

            


    logdf = pd.DataFrame(data = {"actuals":actuals, "preds":preds,"conf":pred_confs, "grade":grades, "track":tracks, "bfsps":bfsps})#, "sp":start_prices })
    
    logdf['correct'] = logdf.apply(lambda x: 1 if x['actuals']==x['preds'] else 0, axis=1)
    logdf['profit'] = logdf.apply(lambda x: 0 if x['bfsps']<1 else x['bfsps']-1  if x['correct'] else -1, axis=1)
    logdf.to_csv('logDFtest.csv')

    table = wandb.Table(dataframe=logdf)

    logdf['count'] = 1
    logdf['eligible_ct'] = logdf.apply(lambda x: 1 if x['bfsps'] > 1 else 0, axis = 1)

    track_df = logdf.groupby('track', as_index=False).sum(numeric_only=True).reset_index()
    
    prices_df = pd.DataFrame(price_dict)


    prices_df['sum_price'] = prices_df.apply(lambda x: sum(x['prices']), axis = 1)
    prices_df = prices_df[prices_df['sum_price']>0]

    prices_flat = [item for sublist in prices_df['prices'].tolist() for item in sublist]
    pred_prices = [item for sublist in prices_df['pred_price'].tolist() for item in sublist]
    onehot_win  = [item for sublist in prices_df['onehot_win'].tolist() for item in sublist]
    flat_margins = [item for sublist in prices_df['margin'].tolist() for item in sublist]
    all_price_df = pd.DataFrame(data={'prices':prices_flat, 'pred_price':pred_prices, 'onehot_win':onehot_win,'split_margin':flat_margins})
    all_price_df = all_price_df[all_price_df['prices']>1]
    all_price_df['imp_prob'] =  all_price_df.apply(lambda x: 1/x['prices'], axis = 1)
    all_price_df['pred_prob'] =  all_price_df.apply(lambda x: 1/x['pred_price'], axis = 1)
    all_price_df['bet amount'] = all_price_df.apply(lambda x: (x['pred_prob']-x['imp_prob'])*100 if (x['pred_prob']>x['imp_prob'])&(1>x['imp_prob']>0) else 0, axis = 1)
    all_price_df['profit'] = all_price_df.apply(lambda x: x['bet amount']*(x['prices']-1)*0.95 if x['onehot_win'] else -1*x['bet amount'], axis = 1)
    all_price_df['flat_profit'] = all_price_df.apply(lambda x: 1*(x['prices']-1)*0.95 if (x['onehot_win'] and x['bet amount']) else -1, axis = 1)
    all_price_df['colour'] = all_price_df.apply(lambda x: "profitz" if x['profit']>0 else ("loss" if x['profit']<0 else ("no bet - win" if x['onehot_win'] else "no bet")), axis=1)


    #print(f"accuray: {correct/total}")
    stats = {"accuracy": correct/total,  "correct": correct, 'profit': logdf[logdf['actuals']==logdf['preds']]['profit'].sum(), 'multibet profit':all_price_df['profit'].sum(), 'multibet outlay':all_price_df['bet amount'].sum() }
    print(stats)
    return all_price_df


def train_regular_L1(model, raceDB:Races, criterion, optimizer,scheduler, config=None):

    
    torch.autograd.set_detect_anomaly(True)

    last = 0
    if config!=None:
        batch_size = config['batch_size']
        batch_before_backwards = config['batch_before_backwards']
        epochs = config['epochs']
    else:
        batch_size = 25
        batch_before_backwards = 5
    len_train = len(raceDB.train_race_ids)
    example_ct = 0  # number of examples seen
    batch_ct = 0
    m = nn.LogSoftmax(dim=1)
    epoch_loss=0
    
    saved_batches = []

    
    #losses = torch.tesnor()
    for epoch in trange(epochs):
        model.train()
        batch_ct = 0
        setup_loss = 1
        for i in trange(0,len_train-batch_size,batch_size):
            last = i
            #print(f"{i=}\n{batch_ct=}, {setup_loss=}, {batch_ct+1%10==0=}")
            if ((batch_ct+1)%batch_before_backwards==0):
                if setup_loss:
                    print("hit here")
                    continue
                else:
                    epoch_loss = torch.mean(epoch_loss)
                    optimizer.zero_grad()
                    epoch_loss.backward()
                    optimizer.step()
                    model.zero_grad()
                    raceDB.detach_all_hidden_states()
                    setup_loss = 1
                    wandb.log({"batch_loss": epoch_loss.item(), "batch_before_backwards": batch_before_backwards}, step = example_ct)
                    
                    #model(saved_batches)
                    #saved_batches = []

            batch_ct += 1

            races_idx = range(last,last+batch_size)
            race = raceDB.get_train_input(races_idx)
            X = race
            #saved_batches.extend(X)
            y = torch.stack([x.classes for x in race])
            w = torch.stack([x.new_weights for x in race])
            _, actual = torch.max(y.data, 1)
            output = model(X)

            
            example_ct +=  batch_size

            loss = criterion(output, y)*w
            if setup_loss:
                epoch_loss = loss
                setup_loss=0

            #epoch_loss = torch.stack([epoch_loss, loss])
            epoch_loss = epoch_loss + loss
            

        #print("finished epoch")
        
        if not setup_loss:
            epoch_loss = torch.mean(epoch_loss)
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()
            model.zero_grad()
            raceDB.detach_all_hidden_states()
            setup_loss = 1
        #wandb.log({"batch_loss": epoch_loss.item(), "batch_before_backwards": batch_before_backwards}, step = example_ct)

        wandb.log({"epoch_loss": torch.mean(epoch_loss), 'epoch':epoch}, step = example_ct)
        # if epoch%3==0:
        acc, loss_val = validate_model(model,raceDB,criterion, 8, example_ct, epoch_loss, batch_ct,epoch,config)
        # compare_model_to_bf(model, raceDB, example_ct)
        train_log(torch.mean(loss), example_ct, epoch)
        epoch_loss = 0  
        scheduler.step(loss_val)
        raceDB.detach_all_hidden_states()
    #print(losses)
    return model

def train_regular_KL(model, raceDB:Races, criterion, optimizer,scheduler, config=None):

    
    torch.autograd.set_detect_anomaly(True)

    last = 0
    if config!=None:
        batch_size = config['batch_size']
        batch_before_backwards = config['batch_before_backwards']
    else:
        batch_size = 25
        batch_before_backwards = 5
    len_train = len(raceDB.train_race_ids)
    example_ct = 0  # number of examples seen
    batch_ct = 0
    m = nn.LogSoftmax(dim=1)
    epoch_loss=0
    
    saved_batches = []

    
    #losses = torch.tesnor()
    for epoch in trange(2000):
        model.train()
        batch_ct = 0
        setup_loss = 1
        for i in trange(0,len_train-batch_size,batch_size):
            last = i
            #print(f"{i=}\n{batch_ct=}, {setup_loss=}, {batch_ct+1%10==0=}")
            if ((batch_ct+1)%batch_before_backwards==0):
                if setup_loss:
                    print("hit here")
                    continue
                else:
                    epoch_loss = torch.mean(epoch_loss)
                    optimizer.zero_grad()
                    epoch_loss.backward()
                    optimizer.step()
                    model.zero_grad()
                    raceDB.detach_all_hidden_states()
                    setup_loss = 1
                    wandb.log({"batch_loss": epoch_loss.item(), "batch_before_backwards": batch_before_backwards}, step = example_ct)
                    
                    #model(saved_batches)
                    #saved_batches = []

            batch_ct += 1

            races_idx = range(last,last+batch_size)
            race = raceDB.get_train_input(races_idx)
            X = race
            #saved_batches.extend(X)
            y = torch.stack([x.classes for x in race])
            w = torch.stack([x.weights for x in race])
            _, actual = torch.max(y.data, 1)
            output = model(X)

            
            example_ct +=  batch_size

            loss = criterion(output, y)*w
            if setup_loss:
                epoch_loss = loss
                setup_loss=0



            #epoch_loss = torch.stack([epoch_loss, loss])
            epoch_loss = epoch_loss + loss
            

        #print("finished epoch")
        setup_loss = 1

        wandb.log({"epoch_loss": torch.mean(epoch_loss), 'epoch':epoch}, step = example_ct)
        if epoch%3==0:
            acc, loss_val = validate_model_KL(model,raceDB,criterion, 8, example_ct, epoch_loss, batch_ct)
        # compare_model_to_bf(model, raceDB, example_ct)
        train_log(torch.mean(loss), example_ct, epoch)
        epoch_loss = 0  
        scheduler.step(loss_val)
        raceDB.detach_all_hidden_states()
    #print(losses)
    return model

def train_regular(model, raceDB:Races, criterion, optimizer,scheduler, config=None):
    # torch.autograd.set_detect_anomaly(True)

    last = 0
    if config!=None:
        batch_size = config['batch_size']
        epochs = config['epochs']
    else:
        batch_size = 25
    len_train = len(raceDB.train_race_ids)
    example_ct = 0  # number of examples seen
    batch_ct = 0
    m = nn.LogSoftmax(dim=1)
    epoch_loss=0
    

    for epoch in trange(epochs):
        model.train()
        batch_ct = 0
        for i in range(0,len_train-batch_size,batch_size):
            optimizer.zero_grad()
            last = i
            batch_ct += 1

            races_idx = range(last,last+batch_size)
            race = raceDB.get_train_input(races_idx)

            X = race
            y = torch.stack([x.classes for x in race])
            w = torch.stack([x.new_win_weight for x in race])

            _, actual = torch.max(y.data, 1)
            output = model(X)
            example_ct +=  batch_size

            loss = criterion(output, y)*w

            epoch_loss = torch.mean(loss)
            
            epoch_loss.backward()
            optimizer.step()
            model.zero_grad()
            wandb.log({"batch_loss": epoch_loss.item()}, step = example_ct)


        
        if epoch%20==0:
            acc, loss_val = validate_model(model,raceDB,criterion, 8, example_ct, epoch_loss, batch_ct,epoch,config)
        train_log(torch.mean(loss), example_ct, epoch)
        epoch_loss = 0  
        scheduler.step(loss_val)

    return model

def train_regular_one_hot(model, raceDB:Races, criterion, optimizer,scheduler, config=None):
    # torch.autograd.set_detect_anomaly(True)

    last = 0
    if config!=None:
        batch_size = config['batch_size']
        batch_before_backwards = config['batch_before_backwards']
    else:
        batch_size = 25
        batch_before_backwards = 5
    len_train = len(raceDB.train_race_ids)
    example_ct = 0  # number of examples seen
    batch_ct = 0
    m = nn.LogSoftmax(dim=1)
    epoch_loss=0
    
    saved_batches = []

    
    #losses = torch.tesnor()
    for epoch in trange(2000):
        model.train()
        batch_ct = 0
        setup_loss = 1
        for i in trange(0,len_train-batch_size,batch_size):
            last = i
            #print(f"{i=}\n{batch_ct=}, {setup_loss=}, {batch_ct+1%10==0=}")
            if ((batch_ct+1)%batch_before_backwards==0):
                if setup_loss:
                    print("hit here")
                    continue
                else:
                    epoch_loss = torch.mean(epoch_loss)
                    optimizer.zero_grad()
                    epoch_loss.backward()
                    optimizer.step()
                    model.zero_grad()
                    raceDB.detach_all_hidden_states()
                    setup_loss = 1
                    wandb.log({"batch_loss": epoch_loss.item(), "batch_before_backwards": batch_before_backwards}, step = example_ct)
                    
                    #model(saved_batches)
                    #saved_batches = []

            batch_ct += 1

            races_idx = range(last,last+batch_size)
            race = raceDB.get_train_input(races_idx)
            X = race
            #saved_batches.extend(X)
            y = torch.stack([x.one_hot_class for x in race])
            w = torch.stack([x.win_price_weight for x in race])
            _, actual = torch.max(y.data, 1)
            output = model(X)

            
            example_ct +=  batch_size

            loss = criterion(output, y)*w
            if setup_loss:
                epoch_loss = loss
                setup_loss=0



            #epoch_loss = torch.stack([epoch_loss, loss])
            epoch_loss = epoch_loss + loss
            

        #print("finished epoch")
        setup_loss = 1

        wandb.log({"epoch_loss": torch.mean(epoch_loss), 'epoch':epoch}, step = example_ct)
        # if epoch%3==0:
        acc, loss_val = validate_model(model,raceDB,criterion, 8, example_ct, epoch_loss, batch_ct, epoch, config)
        # compare_model_to_bf(model, raceDB, example_ct)
        train_log(torch.mean(loss), example_ct, epoch)
        epoch_loss = 0  
        scheduler.step(loss_val)
        raceDB.detach_all_hidden_states()
    #print(losses)
    return model