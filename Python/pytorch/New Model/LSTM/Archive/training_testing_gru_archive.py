import os
import pickle

import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
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



#Testing
def validate_model(model:GRUNet,raceDB:Races,criterion=None, batch_size=None,epoch=10,config=None):
    torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)
    l_sftmax = nn.LogSoftmax(dim=-1)

    # # batch_size= max([int(config['batch_size']/10),250])
    # len_test = len(raceDB.test_race_ids)-batch_size
    # num_batches = len_test/batch_size
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

    len_test = len(raceDB.test_dog_ids)
    test_idx = range(0,len_test)

    # dog_inputs = [[z.full_input for z in inner] for inner in [x for x in raceDB.train_dogs.values()]]
    dogs = [x for x in  raceDB.train_dogs.values()]  #[Dog]
    dog_input = [inner for inner in [x for x in raceDB.get_dog_test(test_idx)]] #[[DogInput]]
   
    for dog in dogs:
        dog.hidden = dog.hidden.to('cuda:0')

    # train = [torch.stack(n,0) for n in [[z.full_input for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]]
    X = pack_sequence([torch.stack(n,0) for n in [[z.full_input for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]], enforce_sorted=False)

    dogs_sorted = [dogs[x] for x in X[3]]
    hidden_in = torch.unsqueeze(torch.stack([x.hidden for x in dogs_sorted]).to('cuda:0'),0)

    margins = [torch.stack(n,0) for n in[[torch.tensor(z.margin).to('cuda:0') for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]]
    Y = pack_sequence(margins, enforce_sorted=False).data.view(-1,1)


    output = model(X,h=hidden_in, train=False) # Shape List[Tensor[Dog]]

    for i,dog in enumerate(dog_input):
        dog_outputs = output[i]
        for j,di in enumerate(dog):
            di.output = dog_outputs[j]

    raceDB.margin_from_dog_to_race()



    test_races = raceDB.get_test_input(test_idx)

    races = {}
    races['race_ids'] = [x.raceid for x in test_races]
    races['raw_margins'] = [x.raw_margins for x in test_races]
    races['output'] = [x.margins for x in test_races]
    races['pred_prob'] = [sft_max(-x) for x in races['output']]
    races['prices'] = [x.prices for x in test_races]
    races['imp_prob'] = [x.implied_prob  for x in test_races]
    races['pred_price'] = [(1/(x)).tolist() for x in races['pred_prob']]
    races['pred_prob'] = [x.tolist() for x in races['pred_prob']]
    races['classes'] = [x.classes for x in test_races]
    races['track'] = [x.track_name for x in test_races]
    races['one_hot_win'] = [x.one_hot_class.tolist() for x in test_races]
    races['date'] = [x.race_date for x in test_races]

    
    # for k,v in races.items():
    #     print(f"{k} len = {len(v)} , type = {type(v)}")

    all_classes = torch.stack(races['classes'])
    all_outputs = torch.stack(races['output'])
    

    races['classes'] = [x.classes.tolist() for x in test_races]
    races['output'] = [x.margins.tolist() for x in test_races]

    _, actual = torch.min(all_classes, 1)

    onehot_win = F.one_hot(actual, num_classes=8)

    
    conf, predicted = torch.min(all_outputs.data, 1)
    correct += (predicted == actual).sum().item()

    accuracy = correct/len(predicted)

    

    basic = wandb.log({"basic_table":wandb.Table(dataframe=pd.DataFrame(races))})


    races['one_hot_win'] = onehot_win.tolist()
    races['track'] = [[x.track_name]*8 for x in test_races]

    prices_flat = [item for sublist in races['prices'] for item in sublist]
    pred_prices = [item for sublist in races['pred_price'] for item in sublist]
    onehot_win  = [item for sublist in races['one_hot_win'] for item in sublist]
    flat_margins = [item for sublist in races['raw_margins'] for item in sublist]
    flat_track = [item for sublist in races['track'] for item in sublist]
    #flat_dogs  = [item for sublist in races['dogID'].tolist() for item in sublist]
    #flat_races = [item for sublist in prices_df['raceID'].tolist() for item in sublist]
    #flat_track = [item for sublist in prices_df['track'].tolist() for item in sublist]
    #flat_date  = [item for sublist in prices_df['date'].tolist() for item in sublist]
    #flat_grade  = [item for sublist in prices_df['grade'].tolist() for item in sublist]
    #flat_loss  = [item for sublist in prices_df['loss'].tolist() for item in sublist]

    all_price_df = pd.DataFrame(data={'track':flat_track, 'prices':prices_flat, 'pred_price':pred_prices, 'onehot_win':onehot_win,'split_margin':flat_margins})
    all_price_df = all_price_df[all_price_df['prices']>1]
    all_price_df['imp_prob'] =  all_price_df.apply(lambda x: 1/x['prices'], axis = 1)
    all_price_df['pred_price'] =  all_price_df['pred_price'].clip(0,100)
    all_price_df['pred_prob'] =  all_price_df.apply(lambda x: 1/x['pred_price'], axis = 1)
    all_price_df['bet amount'] = all_price_df.apply(lambda x: (x['pred_prob']-x['imp_prob'])*100 if (x['pred_prob']>x['imp_prob'])&(1>x['imp_prob']>0) else 0, axis = 1)
    all_price_df['profit'] = all_price_df.apply(lambda x: x['bet amount']*(x['prices']-1)*0.95 if x['onehot_win'] else -1*x['bet amount'], axis = 1)
    all_price_df['win_price'] = all_price_df.apply(lambda x: x['prices'] if (x['onehot_win']) else 0, axis = 1)
    all_price_df['colour'] = all_price_df.apply(lambda x: "profitz" if x['profit']>0 else ("loss" if x['profit']<0 else ("no bet - win" if x['onehot_win'] else "no bet")), axis=1)
    all_price_df['profit < 30'] = all_price_df.apply(lambda x: x['profit'] if x['prices']<30 else 0, axis=1)
    all_price_df['outlay < 30'] = all_price_df.apply(lambda x: x['bet amount'] if x['prices']<30 else 0, axis=1)

    wandb.log({"all_price_df":wandb.Table(dataframe=pd.DataFrame(all_price_df))})

    flat_track_df = all_price_df.groupby('track').sum(numeric_only=True).reset_index()
    flat_track_df['ROI < 30'] = flat_track_df.apply(lambda x: x['profit < 30']/x['outlay < 30'],axis =1)

    wandb.log({'flat_track':wandb.Table(dataframe=flat_track_df)})

    wandb.log({"accuracy": correct/len_test,
                'multibet profit':all_price_df['profit'].sum(),
                'multibet profit < 30':all_price_df['profit < 30'].sum(),
                'multibet profit < 30 sd':all_price_df['profit < 30'].std(),
                'multibet outlay < 30':all_price_df['outlay < 30'].sum(),
                'ROI < 30':all_price_df['profit < 30'].sum()/all_price_df['outlay < 30'].sum(),})

    return accuracy
    


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

def preheat_model(model,raceDB):

    warmup_loss = nn.MSELoss(reduction='mean')
    warmup_optim = optim.Adadelta(model.parameters())
    model.train()   
    warmup_optim.zero_grad()

    # for epoch in trange(warmup_epochs):

    #     output = model(X, warmup=True)
    #     loss = warmup_loss(output, Y)
    #     loss.mean().backward()
    #     warmup_optim.step()
    #     model.zero_grad()
    #     wandb.log({'warmup_loss':loss.mean()})

    #     if epoch%100==0:
    #         model.eval()
    #         output = model(XT, warmup=True)
    #         loss = warmup_loss(output, YT)
    #         wandb.log({'warmup_loss_val':loss.mean()})
    #         model.train()

def train_regular(model, raceDB:Races, criterion, optimizer,scheduler, config=None):
    # torch.autograd.set_detect_anomaly(True)

    last = 0
    if config!=None:
        batch_size = config['batch_size']
        epochs = config['epochs']
    else:
        batch_size = 25
    len_train = len(raceDB.train_dog_ids)
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
        for i in trange(0,len_train,batch_size, leave=False):
            last = i
            batch_ct += 1
            example_ct +=  batch_size

            dogs_idx = range(last,min(last+batch_size,len_train-1))

            

            dogs = [inner[0].dog for inner in [x for x in raceDB.get_dog_train(dogs_idx)]]
            
            #print(dogs)

            train = [torch.stack(n,0) for n in [[z.full_input for z in inner] for inner in [x for x in raceDB.get_dog_train(dogs_idx)]]]
            X = pack_sequence(train, enforce_sorted=False)

            #sorts dogs acroding to packed seq
            dogs_sorted = [dogs[x] for x in X[3]]

            margins = [torch.stack(n,0) for n in[[torch.tensor(z.margin).to('cuda:0') for z in inner] for inner in [x for x in raceDB.get_dog_train(dogs_idx)]]]
            Y = pack_sequence(margins, enforce_sorted=False).data.view(-1,1)

            # w = torch.stack([x.new_win_weight for x in race])
            output,hidden = model(X)
            # print(len(dogs_sorted))
            # print("hidden",hidden)

            #Attaches final hidden state from Train to use at start of Test
            for i,dog in enumerate(dogs_sorted):
                dog.hidden = hidden[0][i]


            # print(output)
            # print(Y)



            loss = criterion(output, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            wandb.log({"batch_loss": loss}, step = example_ct)
        if (epoch%10)==0:
            validate_model(model,raceDB)
    return model

def train_regular_v2(model:GRUNetv2,raceDB:Races, criterion, optimizer,scheduler, config=None):
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
    epoch_loss=0

    warmup_epochs = 2000
    
    saved_batches = []

    dogs_idx = range(0,len_train_dogs)
    print(torch.cuda.memory_allocated())

    optimizer2 = optim.Adadelta(model.parameters())

    dogs = [inner[0].dog for inner in [x for x in raceDB.get_dog_train(dogs_idx)]] #[Dog]
    dog_input = [inner for inner in [x for x in raceDB.get_dog_train(dogs_idx)]] #[[DogInput]]

    print(torch.cuda.memory_allocated())

    torch.cuda.empty_cache()

    train = [torch.stack(n,0) for n in [[z.full_input.to('cpu') for z in inner] for inner in [x for x in raceDB.get_dog_train(dogs_idx)]]]
    # print(torch.cuda.memory_allocated())
    X = pack_sequence(train, enforce_sorted=False).to('cuda:0')
    # print(torch.cuda.memory_allocated())
    #sorts dogs acroding to packed seq
    dogs_sorted = [dogs[x] for x in X[3]] #[Dog]

    len_test = len(raceDB.test_dog_ids)
    test_idx = range(0,len_test)

    test = [torch.stack(n,0) for n in [[z.full_input.to('cpu') for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]]
    print(torch.cuda.memory_allocated())
    XT = pack_sequence(test, enforce_sorted=False).to('cuda:0')

    #Dont actually need to sort input, as unpack sequence returns to order
    dog_input_sorted = [dog_input[x] for x in X[3]] #[[DogInput]]

    margins = [torch.stack(n,0) for n in[[torch.tensor(z.margin).to('cuda:0') for z in inner] for inner in [x for x in raceDB.get_dog_train(dogs_idx)]]]
    # prices = [torch.stack(n,0) for n in[[1/(torch.tensor(z.bfsp).to('cuda:0')+0.001) for z in inner] for inner in [x for x in raceDB.get_dog_train(dogs_idx)]]]
    Y = pack_sequence(margins, enforce_sorted=False).data.view(-1,1)
    # YP = pack_sequence(prices, enforce_sorted=False).data.view(-1,1)

    margins_t = [torch.stack(n,0) for n in[[torch.tensor(z.margin).to('cuda:0') for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]]
    # prices = [torch.stack(n,0) for n in[[1/(torch.tensor(z.bfsp).to('cuda:0')+0.001) for z in inner] for inner in [x for x in raceDB.get_dog_train(dogs_idx)]]]
    YT = pack_sequence(margins_t, enforce_sorted=False).data.view(-1,1)

    warmup_loss = nn.MSELoss(reduction='mean')
    warmup_optim = optim.Adadelta(model.parameters())
    model.train()   
    warmup_optim.zero_grad()

    # for epoch in trange(warmup_epochs):

    #     output = model(X, warmup=True)
    #     loss = warmup_loss(output, Y)
    #     loss.mean().backward()
    #     warmup_optim.step()
    #     model.zero_grad()
    #     wandb.log({'warmup_loss':loss.mean()})

    #     if epoch%100==0:
    #         model.eval()
    #         output = model(XT, warmup=True)
    #         loss = warmup_loss(output, YT)
    #         wandb.log({'warmup_loss_val':loss.mean()})
    #         model.train()

    model.train()

    del Y
    del margins
    del train
    del warmup_optim
    # del loss
    torch.cuda.empty_cache()


    # dogs_idx = range(0,len_train_dogs)
    # dogs = [inner[0].dog for inner in [x for x in raceDB.get_dog_train(dogs_idx)]] #[Dog]
    # dog_input = [inner for inner in [x for x in raceDB.get_dog_train(dogs_idx)]] #[[DogInput]]


    # train = [torch.stack(n,0) for n in [[z.full_input for z in inner] for inner in [x for x in raceDB.get_dog_train(dogs_idx)]]] 
    # X = pack_sequence(train, enforce_sorted=False)


    #sorts dogs acroding to packed seq
    dogs_sorted = [dogs[x] for x in X[3]] #[Dog]

    #Dont actually need to sort input, as unpack sequence returns to order
    # dog_input_sorted = [dog_input[x] for x in X[3]] #[[DogInput]]

    # margins = [torch.stack(n,0) for n in[[torch.tensor(z.margin).to('cuda:0') for z in inner] for inner in [x for x in raceDB.get_dog_train(dogs_idx)]]]
    
    #losses = torch.tesnor()
    for epoch in trange(epochs):
        start_epoch_time = time.perf_counter()
        model.train()
        batch_ct = 0
        batch_ct += 1
        example_ct +=  batch_size

        optimizer.zero_grad()
        # w = torch.stack([x.new_win_weight for x in race])
        output,hidden = model(X)

        t_end_p1 = time.perf_counter()
        print(f'Exec time of t_end_p1 {t_end_p1-start_epoch_time}')

       

        for i,dog in enumerate(dog_input):
            dog_outputs = output[i]
            for j,di in enumerate(dog):
                di.hidden_out = dog_outputs[j]

        t_attache_output = time.perf_counter()
        print(f'Exec time of t_attache_output  {t_attache_output -t_end_p1}')

        print(hidden[0][0])

        for i,dog in enumerate(dogs_sorted):
            dog.hidden = hidden[0][i]

        # print("Margin dog to race")
        raceDB.margin_from_dog_to_race_v2(mode='train')

        t_margin_from_dog_to_race_v2 = time.perf_counter()
        print(f'Exec time of t_attache_output  {t_margin_from_dog_to_race_v2 -t_attache_output}')
        # setup_loss=1
        # for i in trange(0,len_train,batch_size, leave=False):
        # last2=i

        races_idx = range(0, len_train_races)
        race = raceDB.get_train_input(races_idx)

        X2 = torch.stack([r.hidden_in for r in race]) #Input for FFNN
        y = torch.stack([x.classes for x in race])
        w = torch.stack([x.new_win_weight for x in race])
        # w = torch.stack([x.win_price_weight  for x in race])

        output = model(X2, p1=False)
        epoch_loss = criterion(output, y)*w
        epoch_loss.mean().backward()
        optimizer.step()
        model.zero_grad()
        wandb.log({"loss_1": torch.mean(epoch_loss).item()}, step = example_ct)

        validate_model_v2(model,raceDB, criterion=criterion, epoch=epoch)

        # X2 = X2.detach()

        # for i in trange(1,10000):
        #     model.train()
        #     output = model(X2, p1=False)

        #     example_ct +=  batch_size

        #     t_p2 = time.perf_counter()
        #     print(f'Exec time of t_p2  {t_p2-t_margin_from_dog_to_race_v2}')
        #     epoch_loss = criterion(output, y)*w
        #     epoch_loss.mean().backward(retain_graph=True)
        #     optimizer.step()
        #     model.zero_grad()
        #     wandb.log({"loss_2": torch.mean(epoch_loss).item()}, step = example_ct)

        #     t_p2_loss = time.perf_counter()
        #     print(f'Exec time of t_p2_loss  {t_p2_loss-t_p2}')
        #     validate_model_v2(model,raceDB, criterion=criterion, epoch=epoch)

        # if epoch%10==0:
        #     for i,dog in enumerate(dogs_sorted):
        #         dog.hidden = hidden[0][i]
        #     t_v1 =  time.perf_counter()
        #     validate_model_v2(model,raceDB,epoch=epoch)
        #     t_v2 = time.perf_counter()
        #     print(f'Exec time of validate  {t_v2-t_v1}')
        torch.cuda.empty_cache()
    return model

def train_regular_batch(model:GRUNetv2,raceDB:Races, criterion, optimizer,scheduler, config=None):
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

    for epoch in trange(200):
        model.train()

        for i in trange(num_batches):
            dogs = raceDB.batches['dogs'][i]
            train_dog_input = raceDB.batches['train_dog_input'][i]
            batch_races = raceDB.batches['batch_races'][i]
            batch_races_ids = raceDB.batches['batch_races_ids'][i]

            example_ct+=len(batch_races)

            train = [torch.stack(n,0) for n in [[z.full_input.to('cpu') for z in inner] for inner in train_dog_input]]
            X = pack_sequence(train, enforce_sorted=False).to('cuda:0')
            # dogs_sorted = [dogs[x] for x in X[3]]
            
            hidden_in = torch.stack([x.hidden for x in dogs]).to('cuda:0').transpose(0,1)
            output,hidden = model(X, h=hidden_in)

            hidden = hidden.transpose(0,1)


            for i,dog in enumerate(train_dog_input):
                dog_outputs = output[i]
                for j,di in enumerate(dog):
                    di.hidden_out = dog_outputs[j]

            for i,dog in enumerate(dogs):
                dog.hidden = hidden[i]

            raceDB.margin_from_dog_to_race_v2(mode='train')

            race = batch_races

            X2 = torch.stack([r.hidden_in for r in race]) #Input for FFNN
            y = torch.stack([x.classes for x in race])
            w = torch.stack([x.new_win_weight for x in race])

            output = model(X2, p1=False)
            epoch_loss = criterion(output, y)*w
            epoch_loss.mean().backward(retain_graph=True)
            optimizer.step()
            model.zero_grad()
            wandb.log({"loss_1": torch.mean(epoch_loss).item()}, step = example_ct)
            raceDB.detach_hidden(dogs)

        validate_model_v2(model,raceDB, criterion=criterion, epoch=epoch)

        raceDB.reset_hidden()

    return model

def train_regular_v3(model:GRUNetv2,raceDB:Races, criterion, optimizer,scheduler, config=None):
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
                y = torch.stack([x.classes for x in race])
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
            validate_model_v3(model,raceDB, criterion=criterion, epoch=epoch)
        if (epoch+1)%100==0:
            raceDB.create_hidden_states_dict_v2()
            model_saver_wandb(model, optimizer, epoch, 0.1, raceDB.hidden_states_dict_gru_v6 , model_name="long nsw new  22000 RUN")
        # elif (epoch)%3==0:
        #     validate_model_v3(model,raceDB, criterion=criterion, epoch=epoch) 




        raceDB.reset_hidden()

    return model

#Testing
@torch.no_grad()
def validate_model_v3(model:GRUNet,raceDB:Races,criterion=None, batch_size=None,epoch=10,config=None):
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
        y = torch.stack([x.classes for x in race])

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

#Testing
def validate_model_v2(model:GRUNet,raceDB:Races,criterion=None, batch_size=None,epoch=10,config=None):
    torch.autograd.set_detect_anomaly(True)
    sft_max = nn.Softmax(dim=-1)
    l_sftmax = nn.LogSoftmax(dim=-1)

    # # batch_size= max([int(config['batch_size']/10),250])
    # len_test = len(raceDB.test_race_ids)-batch_size
    # num_batches = len_test/batch_size
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
    criterion = nn.CrossEntropyLoss(reduction='none')


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
    
        # for dog in dogs:
        #     dog.hidden = dog.hidden.to('cuda:0')

        train = [torch.stack(n,0) for n in [[z.full_input.to('cpu') for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]]
        X = pack_sequence(train, enforce_sorted=False).to('cuda:0')

        # dogs_sorted = [dogs[x] for x in X[3]]
        hidden_in = torch.stack([x.hidden for x in dogs]).to('cuda:0').transpose(0,1)

        # margins = [torch.stack(n,0) for n in[[torch.tensor(z.margin).to('cuda:0') for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]]
        # Y = pack_sequence(margins, enforce_sorted=False).data.view(-1,1)


        output,hidden = model(X,h=hidden_in) # Shape List[Tensor[Dog]]
        hidden = hidden.transpose(0,1)
        for i,dog in enumerate(dogs):
            dog.hidden_test = hidden[i]

        for i,dog in enumerate(dog_input):
            dog_outputs = output[i]
            # print(f'{i=}, shape of output {len(dog_outputs)} type of output {type(dog_outputs)}')
            for j,di in enumerate(dog):
                di.hidden_out = dog_outputs[j]

        raceDB.margin_from_dog_to_race_v2(mode='test')

        len_test = len(raceDB.test_race_ids)
        test_idx = range(0,len_test)

        race = raceDB.get_test_input(test_idx)

        X = torch.stack([r.hidden_in for r in race]) #Input for FFNN
        y = torch.stack([x.classes for x in race])

        output = model(X, p1=False)
        # print(output)

        _, actual = torch.max(y.data, 1)
        onehot_win = F.one_hot(actual, num_classes=8)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == actual).sum().item()

        softmax_preds = sft_max(output)

        loss = criterion(output, y).mean()



        test_races = raceDB.get_test_input(test_idx)

        races = {}
        races['race_ids'] = [x.raceid for x in test_races]
        races['raw_margins'] = [x.raw_margins for x in test_races]
        # races['output'] = [x.margins for x in test_races]
        races['pred_prob'] = softmax_preds.tolist()
        races['prices'] = [x.prices for x in test_races]
        races['imp_prob'] = [x.implied_prob  for x in test_races]
        races['pred_price'] = (1/softmax_preds).tolist()
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

        all_price_df = pd.DataFrame(data={'flat_dogs':flat_dogs,'flat_races':flat_races,'flat_date':flat_date,'track':flat_track, 'prices':prices_flat, 'pred_price':pred_prices, 'onehot_win':onehot_win,'split_margin':flat_margins})
        all_price_df = all_price_df[all_price_df['prices']>1]
        all_price_df['imp_prob'] =  all_price_df.apply(lambda x: 1/x['prices'], axis = 1)
        all_price_df['pred_price'] =  all_price_df['pred_price'].clip(0,100)
        all_price_df['pred_prob'] =  all_price_df.apply(lambda x: 1/x['pred_price'], axis = 1)
        all_price_df['bet amount'] = all_price_df.apply(lambda x: (x['pred_prob']-x['imp_prob'])*100 if (x['pred_prob']>x['imp_prob'])&(1>x['imp_prob']>0) else 0, axis = 1)
        all_price_df['profit'] = all_price_df.apply(lambda x: x['bet amount']*(x['prices']-1)*0.95 if x['onehot_win'] else -1*x['bet amount'], axis = 1)
        all_price_df['win_price'] = all_price_df.apply(lambda x: x['prices'] if (x['onehot_win']) else 0, axis = 1)
        all_price_df['colour'] = all_price_df.apply(lambda x: "profitz" if x['profit']>0 else ("loss" if x['profit']<0 else ("no bet - win" if x['onehot_win'] else "no bet")), axis=1)
        all_price_df['profit < 30'] = all_price_df.apply(lambda x: x['profit'] if x['prices']<30 else 0, axis=1)
        all_price_df['outlay < 30'] = all_price_df.apply(lambda x: x['bet amount'] if x['prices']<30 else 0, axis=1)



        flat_track_df = all_price_df.groupby('track').sum(numeric_only=True).reset_index()
        flat_track_df['ROI < 30'] = flat_track_df.apply(lambda x: x['profit < 30']/x['outlay < 30'],axis =1)

        all_price_df.to_excel(f'./model_all_price/{wandb.run.name} - all_price_df.xlsx')

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
                    "loss_val":torch.mean(loss)})

        return accuracy