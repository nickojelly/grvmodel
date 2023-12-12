
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
    def __init__(self, dogid, raceid,stats, dog,dog_box, hidden_state, bfsp, sp) -> None:
        self.dogid= dogid
        self.raceid = raceid
        self.stats = stats.to('cuda:0')
        self.dog = dog
        self.gru_cell = hidden_state.float().to('cuda:0')
        self.visited = 0
        self.bfsp = bfsp
        self.gru_cell_out = None

        
        
    def lstm_i(self, hidden_state):
        self.gru_cell = hidden_state
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
        hidden_state = lstm_o
        self.gru_cell_out = lstm_o
        if self.nextrace_id==-1:
            pass
        else:
            self.dog.races[self.nextrace_id].lstm_i(hidden_state) #((lh.detach(), lc.detach())) #DETACH

    def detach_state(self):
        self.gru_cell = self.gru_cell.detach()
            
class Dog:
    def __init__(self, dogid,dog_name, hidden_size, layers) -> None:
        self.dogid = dogid
        self.dog_name = dog_name
        # self.raceids = raceids #possible dictionary of race id keys dog stat outs
        self.lstmcell = 0
        self.layers = layers
        self.hidden_size = hidden_size
        self.l_debug = None
        self.races = {}

    def add_races(self, raceid, racedate, stats,nextraceid, prevraceid, box, bfsp=None, sp=None):
        self.races[raceid] = DogInput(self.dogid, raceid, stats, self, box, torch.randn(self.hidden_size), bfsp, sp) #this is the change
        self.races[raceid].nextrace(nextraceid)
        self.races[raceid].prevrace(prevraceid)

class Race:
    def __init__(self, raceid,trackOHE, dist, classes=None):
        self.raceid = raceid
        self.race_dist = dist.to('cuda:0')
        self.race_track = trackOHE.to('cuda:0')
        self.track_name = None
        if classes!=None:
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

    def add_track_name(self, track_name):
        self.track_name = track_name


    def nn_input(self):
        input = torch.cat([x.stats for x in self.dogs], dim = 0)
        full_input = torch.cat((self.race_dist,self.race_track, input), dim=0).to(device='cuda:0')
        self.full_input = full_input
        return full_input

    def lstm_input(self, pred=False):
        if pred:
            print('pred')
        else:
            l_input = [x.gru_cell for x in self.dogs]
        return l_input

    def lstm_detach(self):
        [x.detach_state for x in self.dogs]

    def list_dogs(self):
        dogs_l = [x for x in self.dogs]
        return dogs_l

    def pass_gru_output(self, hidden_states):
        for i,dog in enumerate(self.dogs):
            hs = hidden_states[i]
            #hs = hs.detach()
            dog.lstm_o(hs) #.clone())

class Races:
    def __init__(self, hidden_size, layers, batch_size = 100) -> None:
        self.racesDict = {}
        self.dogsDict = {}
        self.raceIDs = []
        self.hidden_size = hidden_size
        self.layers = layers
        self.getter = operator.itemgetter(*range(batch_size))

    def add_race(self,raceid:str, trackOHE, dist, classes=None):
        self.racesDict[raceid] = Race(raceid, trackOHE, dist, classes)
        self.raceIDs.append(raceid)

    def add_dog(self,dogid, dog_name):
        if dogid not in self.dogsDict.keys():
            self.dogsDict[dogid] = Dog(dogid,dog_name, self.hidden_size, self.layers)
        # else:
        #     self.dogsDict[dogid] = self.dogsDict[dogid]

    def get_race_input(self, idx) -> Race:
        if len(idx)==1:
            race = self.racesDict[self.raceIDs[idx]]
            print(f"returing race {race}")
            return race
        else:
            raceidx = operator.itemgetter(*idx)
            race_batch_id = raceidx(self.raceIDs)
            races = [self.racesDict[x] for x in race_batch_id] #Returns list of class Race
        
        return races #self.racesDict[raceidx]

    def get_race_classes(self, idx):
        raceidx = self.raceIDs[idx]
        classes = [x for x in self.raceDict[raceidx].classes]
        return classes

    def reset_all_lstm_states(self):
        for race in self.racesDict.values:
            for dog in race.dogs:
                dog.lstmCellc = torch.rand(self.hidden_size)
                dog.lstmCellh = torch.rand(self.hidden_size)
                dog.gru_cell = torch.rand(self.hidden_size)

    def detach_all_hidden_states(self):
        for race in self.racesDict.values():
            for dog in race.dogs:
                dog.gru_cell = dog.gru_cell.detach()

    def create_hidden_states_dict(self):
        self.hidden_states_dict = {}
        for race in self.racesDict.values():
            race_id = race.raceid
            for dog in race.dogs:
                dog_id = dog.dogid
                key = race_id+'_'+dog_id
                val = dog.gru_cell_out
                self.hidden_states_dict[key] = val

    def fill_hidden_states_from_dict(self, hidden_dict):
        for race in self.racesDict.values():
            race_id = race.raceid
            for dog in race.dogs:
                dog_id = dog.dogid
                dog_prev_race_id = dog.prevrace_id
                key = str(dog_prev_race_id)+'_'+dog_id
                try:
                    val = hidden_dict[key]
                    if val != None:
                        dog.gru_cell = val
                    else:
                        dog.gru_cell = torch.rand(self.hidden_size)
                except KeyError as e:
                    print(f'Key error {e}')
                    val = torch.rand(self.hidden_size)
                    dog.gru_cell = val
                    print(f"race in = {dog.gru_cell}")
                print(key,val)

    def to_cuda(self):
        for race in self.racesDict.values():
            race_id = race.raceid
            for dog in race.dogs:
                dog.gru_cell = dog.gru_cell.to(device)

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super(GRUNet, self).__init__()
        self.gru1 = nn.GRUCell(input_size, hidden_size)
        self.gru2 = nn.GRUCell(input_size, hidden_size)
        self.gru3 = nn.GRUCell(input_size, hidden_size)
        self.gru4 = nn.GRUCell(input_size, hidden_size)
        self.gru5 = nn.GRUCell(input_size, hidden_size)
        self.gru6 = nn.GRUCell(input_size, hidden_size)
        self.gru7 = nn.GRUCell(input_size, hidden_size)
        self.gru8 = nn.GRUCell(input_size, hidden_size)
        self.rl1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size * 8, 64)
        self.drop2 = nn.Dropout(dropout)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 8)
        self.hidden_size = hidden_size

    # x represents our data
    def forward(self, race: Race):
        x = torch.stack([r.full_input.float() for r in race])

        # creates list of LSTM data
        hidden_state_in = [list(i) for i in zip(*[r.lstm_input() for r in race])]

        # creates list of tensors for lstm Cells
        hCell = [torch.stack([x for x in y]) for y in hidden_state_in]
        # print(f"{hCell=}")
        # for h in hCell:
        #     print(h._version)
        #     if h.grad_fn != None:
        #         print(h.grad_fn._saved_self)
        #         print(h.grad_fn._saved_other)
        # for h in hCell:
        #     print(h)

        h1 = self.gru1(x, hCell[0])
        h2 = self.gru2(x, hCell[1])
        h3 = self.gru3(x, hCell[2])
        h4 = self.gru4(x, hCell[3])
        h5 = self.gru5(x, hCell[4])
        h6 = self.gru6(x, hCell[5])
        h7 = self.gru7(x, hCell[6])
        h8 = self.gru8(x, hCell[7])

        lstm_list = [
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            h8,
        ]

        hCello = [i for i in zip(*[x for x in lstm_list])]
        # cCello = [i for i in zip(*[x[1] for x in lstm_list])]

        for i, r in enumerate(race):
            r.pass_gru_output(hCello[i])
            # r.lstm_detach()
        xhh = torch.cat((h1, h2, h3, h4, h5, h6, h7, h8), dim=1)
        xr1 = self.rl1(xhh)
        xd1 = self.drop1(xr1)
        xh = self.fc2(xd1)
        xd2 = self.drop2(xh)
        xr2 = self.rl2(xd2)
        xf = self.fc3(xr2)

        output = F.softmax(xf, dim=1)
        #output = nn.LogSoftmax(xf)
        #output = xf
        return output

def build_pred_dataset(data, hidden_size):

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
        raceDB.add_dog(i, j.dog_name.iloc[0])
        j.apply(lambda x: raceDB.dogsDict[i].add_races(x['raceid'], x['date'], torch.Tensor(x['stats']),-1, x['prev_race'], x['box']), axis=1)

    #Fill in races portion
    softmin = nn.Softmin(dim=0)
    races_group = dog_stats_df.groupby(['raceid'])

    null_dog = Dog("nullDog", "no_name", raceDB.hidden_size, raceDB.layers)
    null_dog_i = DogInput("nullDog", "-1", torch.zeros(16), null_dog,0, torch.zeros(raceDB.hidden_size),0,0)
    null_dog_i.nextrace(-1)
    null_dog_i.prevrace(-1)

    null_dog_list = [null_dog] * 8
    #TO FIX LATER PROPER BOX PLACEMENT #FIXED

    races_group = dog_stats_df.groupby(['raceid'])
    for i,j in tqdm(races_group):
    #Track info tensors
        dist = torch.tensor([j.dist.iloc[0]]) 
        trackOHE = torch.tensor(j.trackOHE.iloc[0])

        empty_dog_list = [null_dog_i]*8
        boxes_list = [x for x in j['box']]      
        dog_list = [raceDB.dogsDict[x].races[i] for x in j["dogid"]]

        for n,x in enumerate(boxes_list):
            empty_dog_list[x-1] = dog_list[n]
        
        raceDB.add_race(i,trackOHE,dist)
        
        # List of Dog Input??
        raceDB.racesDict[i].add_dogs(empty_dog_list)
        raceDB.racesDict[i].nn_input()
        raceDB.racesDict[i].add_track_name(j.track_name.iloc[0])
        raceDB.racesDict[i].track_name = j.track_name.iloc[0]
        raceDB.racesDict[i].grade = j.race_grade.iloc[0]

    print(f"number of races = {len(raceDB.racesDict)}, number of unique dogs = {len(raceDB.dogsDict)}")
    return raceDB


def build_dataset(data, hidden_size):

    #Load in pickeled dataframe
    resultsdf = pickle.load(data)
    dog_stats_df = pd.DataFrame(resultsdf)
    dog_stats_df = dog_stats_df.fillna(-1).drop_duplicates(subset=['dogid', 'raceid'])
    dog_stats_df['stats_cuda'] = dog_stats_df.apply(lambda x: torch.tensor(x['stats']), axis =1)
    dog_stats_df['box'] = dog_stats_df['stats'].apply(lambda x: x[0])

    #Created RaceDB
    raceDB = Races(hidden_size, 1)

    num_features_per_dog = len(dog_stats_df['stats'][0])
    print(f"{num_features_per_dog=}")

    #Fill in dog portion:

    dog_stats_group = dog_stats_df.sort_values(['date']).groupby(["dogid"]) 

    for i,j in tqdm(dog_stats_group):
        j["next_race"] = j["raceid"].shift(-1).fillna(-1)
        j["prev_race"] = j["raceid"].shift(1).fillna(-1)
        raceDB.add_dog(i, j.dog_name.iloc[0])
        j.apply(lambda x: raceDB.dogsDict[i].add_races(x['raceid'], x['date'], torch.Tensor(x['stats']),x['next_race'], x['prev_race'], x['box'], x['bfSP'], x['startprice']), axis=1)

    #Fill in races portion
    softmin = nn.Softmin(dim=0)
    races_group = dog_stats_df.groupby(['raceid'])

    null_dog = Dog("nullDog", "no_name", raceDB.hidden_size, raceDB.layers)
    null_dog_i = DogInput("nullDog", "-1", torch.zeros(num_features_per_dog), null_dog,0, torch.zeros(raceDB.hidden_size),0,0)
    null_dog_i.nextrace(-1)
    null_dog_i.prevrace(-1)

    null_dog_list = [null_dog] * 8
    #TO FIX LATER PROPER BOX PLACEMENT #FIXED
    dog_stats_df = dog_stats_df.sort_values('date')
    races_group = dog_stats_df.groupby(['raceid'])
    for i,j in tqdm(races_group):
    #Track info tensors
        dist = torch.tensor([j.dist.iloc[0]]) 
        trackOHE = torch.tensor(j.trackOHE.iloc[0])
        #margins
        empty_dog_list = [null_dog_i]*8
        empty_margin_list = [20]*8
        empty_place_list = [8]*8

        places_list = [x for x in j["place"]]
        boxes_list = [x for x in j['box']]
        margin_list = [x for x in j["margin"]]
        
        dog_list = [raceDB.dogsDict[x].races[i] for x in j["dogid"]]

        #adjustedMargin = [margin_list[x-1] for x in boxes_list]
        for n,x in enumerate(boxes_list):
            empty_margin_list[x-1] = margin_list[n]
            empty_dog_list[x-1] = dog_list[n]
            empty_place_list[x-1] = places_list[n]
        # adjustedMargin = softmin(torch.tensor(empty_margin_list)).to('cuda:0')
        adjustedMargin = softmin(torch.tensor(empty_place_list)).to('cuda:0')
        #adjusted_dog_list = [dog_list[x-1] for x in boxes_list]
        
        raceDB.add_race(i,trackOHE,dist, adjustedMargin)
        
        
        # List of Dog Input??
        raceDB.racesDict[i].add_dogs(empty_dog_list)
        raceDB.racesDict[i].nn_input()
        raceDB.racesDict[i].track_name = j.track_name.iloc[0]
        raceDB.racesDict[i].grade = j.race_grade.iloc[0]

    print(f"number of races = {len(raceDB.racesDict)}, number of unique dogs = {len(raceDB.dogsDict)}")
    return raceDB

def model_saver(model, optimizer, epoch, loss, hidden_state_dict):
    
    pathtofolder = "C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model"
    model_name = wandb.run.name
    isExist = os.path.exists(
        f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/"
    )
    if isExist:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "db":hidden_state_dict,
            },
            f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/{model_name}_{epoch}.pt",
        )
    else:
        print("created path")
        os.makedirs(
            f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "db":hidden_state_dict,
            },
            f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/{model_name}_{epoch}.pt",
        )

#Testing
def validate_model(model,raceDB,criterion, batch_size, example_ct, epoch_loss, batch_ct):
    torch.autograd.set_detect_anomaly(True)
    batch_size=100
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
    with torch.no_grad():
        for i in range(0,123_000,batch_size):
            if ((i+1)%10_000<8000):
                continue
            races_idx = range(last,last+batch_size)
            last = i
            race = raceDB.get_race_input(races_idx)
            #tracks.extend([r.track for r in race])
            X = race
            y = torch.stack([x.classes for x in race])
            output = model(X)
            #print(y)

            _, actual = torch.max(y.data, 1)
            conf, predicted = torch.max(output.data, 1)
            correct += (predicted == actual).sum().item()
            
            total += batch_size
            actuals.extend(actual.tolist())
            preds.extend(predicted.tolist())
            pred_confs.extend(conf.tolist())
            tracks.extend([r.track_name for r in race])
            grades.extend([r.grade for r in race])
            for i,dog_idx in enumerate(actual.tolist()):
                bfsps.append(race[i].dogs[dog_idx].bfsp)
                #start_prices.append(race[i].dogs[dog_idx].sp)

            
            loss = criterion(output, y)
            loss_val += loss

    logdf = pd.DataFrame(data = {"actuals":actuals, "preds":preds,"conf":pred_confs, "grade":grades, "track":tracks, "bfsps":bfsps})#, "sp":start_prices })

    table = wandb.Table(dataframe=logdf)
    try:
        wandb.log({"table_key": table})
    except Exception as e:
        print("e")
    print(f"accuray: {correct/total}")
    wandb.log({"accuracy": correct/total, "loss_val": loss_val/(50_000/batch_size), "correct": correct,"epoch_loss": epoch_loss/batch_ct})

    return correct/total

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)


def train(model, raceDB:Races, criterion, optimizer,scheduler, config=None):
    torch.autograd.set_detect_anomaly(True)
    #return(model)


    last = 0
    batch_size = 500
    example_ct = 0  # number of examples seen
    batch_ct = 0
    m = nn.LogSoftmax(dim=1)

    losses = []

    model.train()
    #epoch_loss = torch.Tensor(0).to(device).requires_grad_(True)
    batch_ct = 0

    setup_loss = 1
    
    for i in trange(0,123_000,batch_size):
        if last>123_000:
            break
        if ((last+1)%10_000>=500):
            print(i)
            #print(f"final epoch before loss = {epoch_loss}")
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()
            model.zero_grad()
            losses.append(epoch_loss)
            raceDB.detach_all_hidden_states()
            setup_loss = 1
            wandb.log({"loss": epoch_loss}, step = example_ct)
            last = i+2000
            break


            # print("in validation area")
            # print(f"{i=}")
            continue
        batch_ct += 1
        
        races_idx = range(last,last+batch_size)
        last = i
        race = raceDB.get_race_input(races_idx)
        #race = [raceDB.racesDict[r],raceDB.racesDict[r]]
        X = race
        # print(f"race id = {X[0].raceid}")

        y = torch.stack([x.classes for x in race])
        _, actual = torch.max(y.data, 1)
        output = model(X)

        
        #print(output,y)
        example_ct +=  batch_size

        loss = criterion(output, y)# *weights
        if setup_loss:
            epoch_loss = loss
            setup_loss=0
        #oss = criterion(m(output), torch.flatten(actual))
        
        #loss.backward(retain_graph=True)  
        #loss.backward()
        #optimizer.step()
        if ((batch_ct + 1) % 25) == 0:
                
                train_log(loss, example_ct,1)

        epoch_loss = epoch_loss + loss
        #output = model(X)

        if ((batch_ct + 1) % 10) == 0:
            continue
            print(f"final epoch before loss = {epoch_loss}")
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()
            model.zero_grad()
            losses.append(epoch_loss)
            raceDB.detach_all_hidden_states()
            setup_loss = 1
            wandb.log({"loss": epoch_loss}, step = example_ct)

        #acc = validate_model(model,raceDB,criterion, 8, example_ct, epoch_loss, batch_ct)
        #print(acc)
        #scheduler.step(acc)
    print(losses)
    return model

def weighted_mse_loss(input, target, weight):
        return (weight * (input - target) ** 2).sum() / weight.sum()


def closure(optimizer, criterion, outs, classes):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(outs, classes)
    loss.backward()
    return loss

def model_pipeline(my_dataset,config=None,prev_model=None, sweep=True):
    if my_dataset:
      dataset = my_dataset    
    else:
      dataset = raceDB
    # tell wandb to get started
    with wandb.init(project="debug", config=config):
      #  access all HPs through wandb.config, so logging matches execution!
      wandb.define_metric("loss", summary="min")
      wandb.define_metric("test_accuracy", summary="max")
      wandb.define_metric("bfprofit", summary="max")
      config = wandb.config
      pprint.pprint(config)
      pprint.pprint(config.epochs)
      print(config)

      model = GRUNet(203,config['hidden_size'])
      # criterion = nn.HuberLoss()
      # criterion = nn.BCEWithLogitsLoss()
      #optimizer = optim.SGD(model.parameters(), lr=0.1)
      #criterion = nn.NLLLoss()
      criterion = nn.CrossEntropyLoss()
      # criterion = nn.SmoothL1Loss(reduction='mean', beta=0.1)

      # optimizer = optim.RMSprop(model.parameters())
      # optimizer = optim.AdamW(model.parameters(), lr=0.001)
      optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',threshold=0.001, patience=100, verbose=True, factor=0.5)
      # optimizer = optim.LBFGS(model.parameters(), lr=0.001)
      model = model.to(device)
      # optimizer = optimizer.to(device)
      print(model)

      # and use them to train the model
      train(model, dataset, criterion, optimizer, scheduler, config)
      dataset.create_hidden_states_dict()
      # if sweep:
    #   raceDB.reset_all_lstm_states
    


    # and test its final performance
    #test(model, test_loader)

    return (model,dataset)

if __name__=="__main__":
    # with torch.cuda.profiler.profile() as prof:
    
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,with_stack=True, profile_memory=True) as prof:   
    #         with record_function("model_training"):
    #             print('he')

    #             (model,dataset) = model_pipeline(raceDB,config=wandb_config_static)

    os.getcwd()
    os.chdir(r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\DATA")
    dog_stats_file = open( 'dog_stats_df_FASTTRACK.npy', 'rb')
    hidden_size = 32
    raceDB = build_dataset(dog_stats_file, hidden_size)
    wandb_config_static = {'hidden_size':hidden_size,'batch_size': 500, 'dropout': 0.3, 'epochs': 1000, 'f1_layer_size': 256, 'f2_layer_size': 64 , 'learning_rate': 0.0001, 'loss': 'L1', 'l1_beta':0.1,  'num_layers': 2, 'optimizer': 'adamW', 'validation_split': 0.1}

    with torch.profiler.profile() as profiler:
        pass

    with torch.cuda.profiler.profile() as prof:

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,with_stack=True, profile_memory=True) as prof:   
            with record_function("model_training"):
                (model,dataset) = model_pipeline(raceDB,config=wandb_config_static)
    prof.export_chrome_trace("trace_new.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")

