#NO WANDB
import pickle
import pandas as pd
import os

import torch
with torch.profiler.profile() as profiler:
        pass


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb
from torch.utils.data.sampler import SubsetRandomSampler
import pprint
import matplotlib.pyplot as plt
import torch

from sklearn.preprocessing import MinMaxScaler
import math
from torch.profiler import profile, record_function, ProfilerActivity

from operator import itemgetter
import operator


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


class DogInput:
    def __init__(self, dogid, raceid,stats, dog,dog_box, lstmCellh,lstmCellc) -> None:
        self.dogid= dogid
        self.raceid = raceid
        self.stats = stats.to('cuda:0')
        self.dog = dog
        self.lstmCellh = lstmCellh.float().to('cuda:0')
        self.lstmCellc = lstmCellc.float().to('cuda:0')
        self.visited = 0
        
        
    def lstm_i(self, lstmInput):
        (self.lstmCellh,self.lstmCellc) = lstmInput
        # self.lstmCellh=self.lstmCellh.to(device)
        # self.lstmCellc=self.lstmCellc.to(device)
        self.visited = self.visited + 1
        # if self.visited>1:
        #     print("FOUND LEAK")
        #     sasdfasd

    def nextrace(self, raceid):
        self.nextrace_id = raceid

    def prevrace(self, raceid):
        self.prevrace_id = raceid

    def lstm_o(self, lstm_o):
        # print(lstm_o[0]._version)
        (lh,lc) = lstm_o
        if self.nextrace_id==-1:
            pass
        else:
            self.dog.races[self.nextrace_id].lstm_i((lh.detach(), lc.detach())) #DETACH
            


class Dog:
    def __init__(self, dogid, hidden_size, layers) -> None:
        self.dogid = dogid
        # self.raceids = raceids #possible dictionary of race id keys dog stat outs
        self.lstmcell = 0
        self.layers = layers
        self.hidden_size = hidden_size
        self.l_debug = None
        self.races = {}

    def add_races(self, raceid, racedate, stats,nextraceid, prevraceid, box):
        self.races[raceid] = DogInput(self.dogid, raceid, stats, self, box, torch.randn(self.hidden_size).clone(),torch.randn(self.hidden_size).clone()) #this is the change
        self.races[raceid].nextrace(nextraceid)
        self.races[raceid].prevrace(prevraceid)

class Race:
    def __init__(self, raceid,trackOHE, dist, classes):
        self.raceid = raceid
        self.race_dist = dist.to('cuda:0')
        self.race_track = trackOHE.to('cuda:0')
        self.classes =  classes.to('cuda:0')

    def add_dogs(self, dogs_list:DogInput):
        self.dog1 = dogs_list[0]
        self.dog2 = dogs_list[1]
        self.dog3 = dogs_list[2]
        self.dog4 = dogs_list[3]
        self.dog5 = dogs_list[4]
        self.dog6 = dogs_list[5]
        self.dog7 = dogs_list[6]
        self.dog8 = dogs_list[7]
        self.dogs = dogs_list

    def nn_input(self):
        input = torch.cat([x.stats for x in self.dogs], dim = 0)
        full_input = torch.cat((self.race_dist,self.race_track, input), dim=0).to(device='cuda:0')
        self.full_input = full_input
        return full_input

    def lstm_input(self):
        
        l_input = [(x.lstmCellh,x.lstmCellc) for x in self.dogs]
        return l_input

    def list_dogs(self):
        dogs_l = [x for x in self.dogs]
        return dogs_l

    def pass_lstm_output(self, lstm_h, lstm_c):
        for i,dog in enumerate(self.dogs):
            
            lh = lstm_h[i]
            lc = lstm_c[i]
            lh,lc = lh.detach(), lc.detach() # .clone()
            dog.lstm_o((lh,lc))
            dog.dog.l_debug = (lh,lc)
        # zipped_lstm = zip(self.dogs,lstms)
        # [x.lstm_o(y) for x,y in zipped_lstm]

class Races:
    def __init__(self, hidden_size, layers, batch_size = 100) -> None:
        self.racesDict = {}
        self.dogsDict = {}
        self.raceIDs = []
        self.hidden_size = hidden_size
        self.layers = layers
        self.getter = operator.itemgetter(*range(batch_size))

    def add_race(self,raceid:str, trackOHE, dist, classes):
        self.racesDict[raceid] = Race(raceid, trackOHE, dist, classes)
        self.raceIDs.append(raceid)

    def add_dog(self,dogid):
        if dogid not in self.dogsDict.keys():
            self.dogsDict[dogid] = Dog(dogid, self.hidden_size, self.layers)
        else:
            self.dogsDict[dogid] = self.dogsDict[dogid]

    def get_race_input(self, idx) -> Race:
        raceidx = operator.itemgetter(*idx)
        #raceidx  = self.getter(idx)
        race_batch_id = raceidx(self.raceIDs)

        #race_getter = operator.itemgetter(*raceidx)

        races = [self.racesDict[x] for x in race_batch_id] #Returns list of class Race
        

        # raceidx = self.raceIDs[idx]
        #input = torch.cat([x.stats for x in races.dogs.values()], dim = 0)
        #full_input = torch.cat((self.racesDict[raceidx].race_dist,self.racesDict[raceidx].race_track, input), dim=0 )
        # dogs = [x for x in self.racesDict[raceidx].dogs]
        
        return races #self.racesDict[raceidx]

    def get_race_classes(self, idx):
        raceidx = self.raceIDs[idx]
        classes = [x for x in self.raceDict[raceidx].classes]
        return classes


class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(input_size, hidden_size)
        self.lstm3 = nn.LSTMCell(input_size, hidden_size)
        self.lstm4 = nn.LSTMCell(input_size, hidden_size)
        self.lstm5 = nn.LSTMCell(input_size, hidden_size)
        self.lstm6 = nn.LSTMCell(input_size, hidden_size)
        self.lstm7 = nn.LSTMCell(input_size, hidden_size)
        self.lstm8 = nn.LSTMCell(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 8, 64)
        self.fc3 = nn.Linear(64, 8)
        self.hidden_size = hidden_size

    # x represents our data
    def forward(self, race: Race):
        #x = race.nn_input().float().to('cuda:0')
        x = torch.stack([r.full_input.float() for r in race])

        #creates list of LSTM data 
        lstm_ins = [list(i) for i in zip(*[r.lstm_input() for r in race])]

        # creates list of tensors for lstm Cells
        hCell = [torch.stack([x[0] for x in y]) for y in lstm_ins]
        cCell = [torch.stack([x[1] for x in y]) for y in lstm_ins]

        (h1, c1) = self.lstm1(x, (hCell[0], cCell[0]))
        (h2, c2) = self.lstm2(x, (hCell[1], cCell[1]))
        (h3, c3) = self.lstm3(x, (hCell[2], cCell[2]))
        (h4, c4) = self.lstm4(x, (hCell[3], cCell[3]))
        (h5, c5) = self.lstm5(x, (hCell[4], cCell[4]))
        (h6, c6) = self.lstm6(x, (hCell[5], cCell[5]))
        (h7, c7) = self.lstm7(x, (hCell[6], cCell[6]))
        (h8, c8) = self.lstm8(x, (hCell[7], cCell[7]))

        lstm_list = [
            (h1, c1),
            (h2, c2),
            (h3, c3),
            (h4, c4),
            (h5, c5),
            (h6, c6),
            (h7, c7),
            (h8, c8)
        ]

        hCello = [i for i in zip(*[x[0] for x in lstm_list])]
        cCello = [i for i in zip(*[x[1] for x in lstm_list])]

        for i,r in enumerate(race):
            r.pass_lstm_output(hCello[i],cCello[i])
        xhh = torch.cat((h1,h2, h3, h4, h5, h6, h7, h8), dim=1)# .clone()
        xh = self.fc2(xhh)
        xf = self.fc3(xh)

        output = F.softmax(xf, dim=0)
        return output


def build_dataset(data, hidden_size, rebuild_data=True):
    if rebuild_data:

        #Load in pickeled dataframe
        resultsdf = pickle.load(data)
        dog_stats_df = pd.DataFrame(resultsdf)
        dog_stats_df = dog_stats_df.fillna(-1).drop_duplicates(subset=['dogid', 'raceid'])
        dog_stats_df['stats_cuda'] = dog_stats_df.apply(lambda x: torch.tensor(x['stats']), axis =1)
        dog_stats_df['box'] = dog_stats_df['stats'].apply(lambda x: x[0])

        #Created RaceDB
        raceDB = Races(hidden_size, 1)

        #Fill in dog portion:

        dog_stats_group = dog_stats_df.sort_values(['date']).groupby(["dogid"])

        for i,j in tqdm(dog_stats_group):
            j["next_race"] = j["raceid"].shift(-1).fillna(-1)
            j["prev_race"] = j["raceid"].shift(1).fillna(-1)
            raceDB.add_dog(i)
            j.apply(lambda x: raceDB.dogsDict[i].add_races(x['raceid'], x['date'], torch.Tensor(x['stats']),x['next_race'], x['prev_race'], x['box']), axis=1)

        #Fill in races portion
        softmin = nn.Softmin(dim=0)

        races_group = dog_stats_df.groupby(['raceid'])

        null_dog = Dog("nullDog", raceDB.hidden_size, raceDB.layers)
        null_dog_i = DogInput("nullDog", "-1", torch.zeros(16), null_dog,0, torch.zeros(raceDB.hidden_size), torch.zeros(raceDB.hidden_size))
        null_dog_i.nextrace(-1)
        null_dog_i.prevrace(-1)

        null_dog_list = [null_dog] * 8
        #TO FIX LATER PROPER BOX PLACEMENT #FIXED

        races_group = dog_stats_df.groupby(['raceid'])
        for i,j in tqdm(races_group):
        #Track info tensors
            dist = torch.tensor([j.dist.iloc[0]]) 
            trackOHE = torch.tensor(j.trackOHE.iloc[0])
            #margins
            empty_dog_list = [null_dog_i]*8
            empty_margin_list = [100]*8
            boxes_list = [x for x in j['box']]
            margin_list = [x for x in j["place"]]
            dog_list = [raceDB.dogsDict[x].races[i] for x in j["dogid"]]

            #adjustedMargin = [margin_list[x-1] for x in boxes_list]
            for n,x in enumerate(boxes_list):
                empty_margin_list[x-1] = margin_list[n]
                empty_dog_list[x-1] = dog_list[n]
            adjustedMargin = softmin(torch.tensor(empty_margin_list)).to('cuda:0')

            #adjusted_dog_list = [dog_list[x-1] for x in boxes_list]
            
            raceDB.add_race(i,trackOHE,dist, adjustedMargin)
            
            
            # List of Dog Input??
            raceDB.racesDict[i].add_dogs(empty_dog_list)
            raceDB.racesDict[i].nn_input()
            # with open(r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\DATA\raceDB.npy", "wb") as fp:   #Pickling
    
            #     pickle.dump(raceDB, fp)
    
    else:
        pass
        #file = open(r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\DATA\raceDB.npy", "rb")
        #raceDB = pickle.load(file)


    return raceDB


#Testing
def validate_model(model,raceDB,criterion, batch_size, example_ct):
    torch.autograd.set_detect_anomaly(True)
    list_t = [] 
    last = 0
    loss_val = 0 
    correct = 0
    total = 0
    with torch.no_grad():
        for i in trange(60000,70000,batch_size):   
            races_idx = range(last,last+batch_size)
            last = i
            race = raceDB.get_race_input(races_idx)
            X = race
            y = torch.stack([x.classes for x in race])
            output = model(X)
            #print(y)
            _, actual = torch.max(y.data, 1)
            _, predicted = torch.max(output.data, 1)
            #print(predicted)
            #print(actual)
            correct += (predicted == actual).sum().item()
            total +=10



            loss = criterion(output, y)
            #optimizer.zero_grad()
            #newnet.zero_grad()
            #loss.backward(retain_graph=True)  
            #optimizer.step()
            #if i %5000 == 0:
            #    print(loss)
        # optimizer.step() 
        #print(loss)
    print(f"accuray: {correct/total}")
    wandb.log({"accuracy": correct/total, "loss": loss}, step=example_ct)

def train(model, raceDB, criterion, optimizer, config=None):
    torch.autograd.set_detect_anomaly(True)

    last = 0
    batch_size = 100
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in range(1): 
        for i in trange(0,6000,batch_size):
            batch_ct += 1   
            races_idx = range(last,last+batch_size)
            last = i
            race = raceDB.get_race_input(races_idx)
            X = race

            y = torch.stack([x.classes for x in race])
            output = model(X)
            example_ct +=  batch_size
            batch_ct += 1

            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)  
            optimizer.step()
            if ((batch_ct + 1) % 25) == 0:
                    pass
                    #train_log(loss, example_ct, epoch)

        print(loss)
        #validate_model(model,raceDB,criterion, 8, example_ct)

    return model



def model_pipeline(my_dataset,config=None,prev_model=None):
    dataset = my_dataset
    # tell wandb to get started
    #config = wandb.config
    #pprint.pprint(config)
    #pprint.pprint(config.epochs)
    print(config)

    model = Net(144,64)
    # criterion = nn.HuberLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # make the model, data, and optimization problem
    #model, train_loader, test_loader, criterion, optimizer = make(config, dataset)


    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model = model.to(device)
    #optimizer = optimizer.to(device)
    print(model)

    # and use them to train the model
    train(model, dataset, criterion, optimizer, config)

    # and test its final performance
    #test(model, test_loader)

    return model


if __name__ == "__main__":
    os.getcwd()
    #os.chdir(r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\DATA")
    dog_stats_file = open( r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\DATA\dog_stats_df.npy", 'rb')
    hidden_size = 64
    raceDB = build_dataset(dog_stats_file, hidden_size)
    wandb_config_static = {'hidden_size':hidden_size,'batch_size': 360, 'dropout': 0.3, 'epochs': 1, 'f1_layer_size': 256, 'f2_layer_size': 64 , 'learning_rate': 0.00001, 'loss': 'L1', 'l1_beta':0.1,  'num_layers': 2, 'optimizer': 'adamW', 'validation_split': 0.1}
    print("finished")
    CUDA_LAUNCH_BLOCKING=1
    import torch
    with torch.profiler.profile() as profiler:
            pass
    import torch
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,with_stack=True, profile_memory=True) as prof:
        with record_function("model_training"):
            print(f"starting {os.getpid()}")

            model = model_pipeline(raceDB,config=wandb_config_static)
    prof.export_chrome_trace("trace_new.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")

    print("done")