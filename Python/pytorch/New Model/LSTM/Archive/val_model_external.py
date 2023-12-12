from rnn_classes import *
import pickle
import pandas as pd
import wandb
import os


#Testing
def validate_model(model:GRUNet,raceDB:Races,model_name):
    torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)
    l_sftmax = nn.LogSoftmax(dim=-1)

    loss_fun = nn.CrossEntropyLoss()

    batch_size= 250
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
    price_dict['track'] = []
    price_dict['date'] = []
    price_dict['grade'] = []
    race_ids = []


    with torch.no_grad():
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

            
            loss = loss_fun(output, y).detach()

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
            #print([(1/(x+(-7**10))).tolist() for x in torch.exp(output)])
            price_dict['pred_price'].extend([(1/(x)).tolist() for x in softmax_preds])
            price_dict['margin'].extend([x.raw_margins for x in race])
            price_dict['onehot_win'].extend(onehot_win.tolist())
            price_dict['raceID'].extend([[x.raceid]*8 for x in race])
            # price_dict['raceID'].extend([[x.raceid]*8 for x in race])
            price_dict['dogID'].extend([x.list_dog_ids() for x in race])
            price_dict['track'].extend([[x.track_name]*8 for x in race])
            price_dict['date'].extend([[x.race_date]*8 for x in race])
            price_dict['grade'].extend([[x.grade]*8 for x in race])

            

        loss_list = []

        # #print("start loss calc")
        # for i,l in enumerate(loss_l):
        #     for j in range(0,7):
        #         loss_list.append([preds_l[i][j],margins_l[i][j],loss_t[i][j],l[j],pred_sftmax[i][j],margins_prob[i][j], raw_margins[i][j], raw_places[i][j]])

    # loss_df = pd.DataFrame(loss_list, columns=['Preds','Margins','Indiviual Losses','Losses','softmaxPreds','Softmax Margins','Raw margins', 'Raw places'])
    # loss_table = wandb.Table(dataframe=loss_df)


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
    # flat_loss  = [item for sublist in prices_df['loss'].tolist() for item in sublist]
    all_price_df = pd.DataFrame(data={'flat_dogs':flat_dogs,'flat_races':flat_races,'prices':prices_flat, 'pred_price':pred_prices, 'onehot_win':onehot_win,'split_margin':flat_margins,'flat_track':flat_track,'flat_date':flat_date,'flat_grade':flat_grade})
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

    flat_track_df = all_price_df.groupby('flat_track').sum(numeric_only=True).reset_index()
    flat_track_df['ROI < 30'] = flat_track_df.apply(lambda x: x['profit < 30']/x['outlay < 30'],axis =1)
    # flat_track_df['loss avg'] = flat_track_df.apply(lambda x: x['flat_loss']/x['onehot_win'], axis=1)
    flat_track_df['profit_captured'] = flat_track_df.apply(lambda x: x['flat_profit']/x['win_price'], axis=1)
    all_price_df.to_pickle('all_price_df_nz.npy')

    all_price_df.to_excel(f'./model_all_price/{model_name} - TESTING - all_price_df.xlsx')

    return correct/total, all_price_df