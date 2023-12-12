import torch.nn as nn
import torch
from operator import itemgetter
import operator
from random import randint
#from tqdm.notebook import tqdm, trange
from tqdm import tqdm, trange
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence,pad_sequence, unpack_sequence, unpad_sequence
import os
import datetime
from goto_conversion import goto_conversion

print(os.getcwd())



class DogInput:
    def __init__(self, dogid, raceid,stats, dog,dog_box, hidden_state, bfsp, sp, margin=None, hidden_size=64) -> None:
        self.dogid= dogid
        self.raceid = raceid
        self.stats = stats.to('cuda:0')
        self.dog = dog
        self.gru_cell = hidden_state.float().to('cuda:0')
        self.visited = 0
        self.bfsp = bfsp
        self.gru_cell_out = None
        self.lstmCellh = None
        self.lstmCellc = None
        self.hidden = (-torch.ones(hidden_size)).to('cuda:0')
        self.hidden_out = (-torch.ones(hidden_size)).to('cuda:0')
        self.cell_out = None
        self.box = dog_box
        self.gru_filled = 0
        self.margin = margin
        self.output = torch.tensor([100.1]).to('cuda:0')

    def gru_i(self, hidden_state):
        self.gru_cell = hidden_state
        self.visited = self.visited + 1
        self.gru_filled = 1


    def lstm_2(self, hidden_state):
        self.gru_cell = hidden_state
        self.visited = self.visited + 1

    def nextrace(self, raceid):
        self.nextrace_id = raceid

    def prevrace(self, raceid):
        self.prevrace_id = raceid

    def gru_o(self, gru_o):
        # print(lstm_o[0]._version)
        hidden_state = gru_o
        self.gru_cell_out = gru_o
        if self.nextrace_id==-1:
            pass
        else:
            self.dog.races[self.nextrace_id].gru_i(hidden_state) 

    def lstm_o(self, hidden):
        # print(lstm_o[0]._version)
        hidden_state = hidden[0]
        cell_state = hidden[1]

        self.hidden_out = hidden_state
        self.cell_out = cell_state
        if self.nextrace_id==-1:
            pass
        else:
            self.dog.races[self.nextrace_id].lstmCellh =hidden_state
            self.dog.races[self.nextrace_id].lstmCellc = cell_state

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
        self.hidden = torch.zeros(2, hidden_size).to('cuda:0')
        self.hidden_filled = 0
        self.l_debug = None
        self.races = {} #:dict[str,DogInput]
        self.race_train = [] #[DogInput]
        self.races_test = [] #[DogInput]

    def add_races(self, raceid, racedate, stats,nextraceid, prevraceid, box,margin=None, bfsp=None, sp=None):
        self.races[raceid] = DogInput(self.dogid, raceid, stats, self, box, torch.ones(self.hidden_size), bfsp, sp, margin, self.hidden_size) #this is the change
        self.races[raceid].nextrace(nextraceid)
        self.races[raceid].prevrace(prevraceid)

class Race:
    def __init__(self, raceid,trackOHE, dist, classes=None):
        self.raceid = raceid
        self.race_dist = dist.to('cuda:0')
        self.race_track = trackOHE.to('cuda:0')
        self.track_name = None
        self.raw_margins = None
        self.raw_places = None
        self.raw_prices = None
        
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
        full_input = torch.cat((self.race_dist,self.race_track, input), dim=0)
        self.full_input = full_input
        return full_input

    def gru_input(self, pred=False):
        if pred:
            print('pred')
        else:
            l_input = [x.gru_cell for x in self.dogs]
        return l_input

    def lstm_input(self, pred=False):
        if pred:
            print('pred')
        else:
           l_input = [(x.lstmCellh,x.lstmCellc) for x in self.dogs]
        return l_input

    def lstm_detach(self):
        [x.detach_state for x in self.dogs]

    def list_dog_ids(self):
        dogs_l = [x.dog.dogid for x in self.dogs]
        return dogs_l
    
    def list_dog_names(self):
        dogs_l = [x.dog.dog_name for x in self.dogs]
        return dogs_l


    def pass_lstm_output(self, lstm_h, lstm_c):
        for i,dog in enumerate(self.dogs):
            lh = lstm_h[i]
            lc = lstm_c[i]
            lh,lc = lh, lc
            # lh,lc = lh.clone(), lc.clone()
            dog.lstm_o((lh,lc))

    def pass_gru_output(self, hidden_states):
        for i,dog in enumerate(self.dogs):
            hs = hidden_states[i]
            #hs = hs.detach()
            dog.gru_o(hs) #.clone())
            
    def add_weights(self, weights):
        self.weights = weights

    def add_win_weight(self):
        pass

    

class Races:
    def __init__(self, hidden_size:int, layers:int, batch_size = 100, device='cuda:0') -> None:
        self.racesDict = {}#  dict[str,Race]
        self.dogsDict = {}# dict[str,Dog]
        self.raceIDs = []
        self.dog_ids = []
        self.hidden_size = hidden_size
        self.layers = layers
        self.getter = operator.itemgetter(*range(batch_size))
        self.device = device

    def add_race(self,raceid:str, trackOHE, dist, classes=None):
        self.racesDict[raceid] = Race(raceid, trackOHE, dist, classes)
        self.raceIDs.append(raceid)

    def add_dog(self,dogid, dog_name):
        # if dogid not in self.dogsDict.keys():
        self.dogsDict[dogid] = Dog(dogid,dog_name, self.hidden_size, self.layers)

    def create_test_split(self):
        self.train_race_ids = []
        self.test_race_ids = []
        for i,r in enumerate(self.raceIDs):
            if i%10>(8):
                self.test_race_ids.append(r)
            else:
                self.train_race_ids.append(r)

    def create_test_split_date(self, start_date, val_date = "2023-07-01"):
        self.train_race_ids = []
        self.test_race_ids = []
        self.val_race_ids = []
        start_date = datetime.datetime.strptime(start_date[0:10], "%Y-%m-%d").date()
        val_race_date = datetime.datetime.strptime(val_date, "%Y-%m-%d").date()
        test,train,val = 0,0,0
        for i,r in enumerate(self.raceIDs):
            race_date = self.racesDict[r].race_date
            if race_date>val_race_date:
                self.val_race_ids.append(r)
                val += 1
            elif race_date>start_date:
                self.test_race_ids.append(r)
                test += 1
            else:
                self.train_race_ids.append(r)
                train += 1
        self.train_race_ids_set = set(self.train_race_ids)
        self.test_race_ids_set = set(self.test_race_ids)
        self.val_race_ids_set = set(self.val_race_ids)

    
        print(f"Train examples {train}, Test examples {test}, Val examples {val}")

    def create_dogs_test_split_date(self):
        self.train_dogs = {}
        self.test_dogs = {}
        self.val_dogs = {}
        self.train_dog_ids = []
        self.test_dog_ids = []
        self.val_dog_ids = []
        for i in tqdm(self.dog_ids):
            dog = self.dogsDict[i]
            train = [dog.races[x] for x in dog.races.keys() if x in self.train_race_ids_set]
            test =  [dog.races[x] for x in dog.races.keys() if x in self.test_race_ids_set]
            val = [dog.races[x] for x in dog.races.keys() if x in self.val_race_ids_set]
            if train:
                dog.train = train
                self.train_dogs[i] = dog
                self.train_dog_ids.append(i)
            if test:
                dog.test = test
                self.test_dogs[i] = dog
                self.test_dog_ids.append(i)
            if val:
                dog.val = val
                self.val_dogs[i] = dog
                self.val_dog_ids.append(i)

    def get_race_input(self, idx, Train=True) -> Race:
        if len(idx)==1:
            race = self.racesDict[self.raceIDs[idx]]
            print(f"returing race {race}")
            return race
        else:
            raceidx = operator.itemgetter(*idx)
            race_batch_id = raceidx(self.raceIDs)
            races = [self.racesDict[x] for x in race_batch_id] #Returns list of class Race
        
        return races #self.racesDict[raceidx]

    def get_train_input(self, idx, Train=True) -> Race:
        raceidx = operator.itemgetter(*idx)
        race_batch_id = raceidx(self.train_race_ids)
        races = [self.racesDict[x] for x in race_batch_id] #Returns list of class Race
        return races #self.racesDict[raceidx]

    def get_test_input(self, idx, Train=True) -> Race:
        raceidx = operator.itemgetter(*idx)
        race_batch_id = raceidx(self.test_race_ids)
        races = [self.racesDict[x] for x in race_batch_id] #Returns list of class Race
        return races #self.racesDict[raceidx]

    def get_val_input(self, idx, Train=True) -> Race:
        raceidx = operator.itemgetter(*idx)
        race_batch_id = raceidx(self.val_race_ids)
        races = [self.racesDict[x] for x in race_batch_id] #Returns list of class Race
        return races #self.racesDict[raceidx]

    def get_race_classes(self, idx):
        raceidx = self.raceIDs[idx]
        classes = [x for x in self.raceDict[raceidx].classes]
        return classes

    def create_hidden_states_dict_v2(self):
        self.hidden_states_dict_gru_v6 = {}
        self.train_hidden_dict = {}
        for dog_id in self.test_dog_ids:
            dog = self.dogsDict[dog_id]
            dog_id = dog.dogid
            try:
                dog.test
                hidden_test = dog.hidden_test
                last_race_test = dog.test[-1].race.race_date
                self.hidden_states_dict_gru_v6[dog_id] = (hidden_test,last_race_test)
            except Exception as e:
                pass

            try:
                hidden_train = dog.hidden
                last_race_train = dog.train[-1].race.race_date
                self.train_hidden_dict[dog_id] = (hidden_train,last_race_train)
            except Exception as e:
                pass

    def fill_hidden_states_dict_v2(self, hidden_dict):
        filled, empty, null_dog = 0, 0, 0
        for dog in tqdm(self.dogsDict.values()):
            dog_id = dog.dogid
            try:
                val = hidden_dict[dog_id]
                if type(val)==tuple:
                    hidden,last_race_date = val
                    dog.hidden= hidden
                    dog.last_race_date = last_race_date
                    dog.hidden_filled = 1
                elif val != None:
                    dog.hidden = val
                    dog.hidden_filled = 1
                    # print(f"{dog,dog.hidden_filled,dog_id=}")
                else:
                    dog.hidden_filled = 0
                filled += 1
            except KeyError as e:
                empty += 1
                dog.hidden_filled = 0
                dog.last_race_date = "NA"
                # print(f"Empty: {dog_id} {dog.dog_name}")

        print(f"{filled =}\n{empty  =}\n{filled/(filled+empty)}{null_dog=}")

    def fill_hidden_states_from_dict(self, hidden_dict):
        filled, empty, null_dog = 0, 0, 0
        for race in tqdm(self.racesDict.values()):
            race_id = race.raceid
            for dog in race.dogs:
                dog_id = dog.dogid
                if dog_id==-1:
                    null_dog +=1
                    continue
                dog_prev_race_id = dog.prevrace_id
                key = str(dog_prev_race_id)+'_'+dog_id
                try:
                    val = hidden_dict[key]
                    if val != None:
                        dog.gru_cell = val
                    else:
                        dog.gru_cell = torch.ones(self.hidden_size) # torch.rand(self.hidden_size)
                    filled += 1
                except KeyError as e:
                    empty +=1
                    val = torch.ones(self.hidden_size) # torch.rand(self.hidden_size)
                    dog.gru_cell = val
        print(f"{filled =}\n{empty  =}\n{filled/(filled+empty)}{null_dog=}")

    def to_cuda(self):
        for race in self.racesDict.values():
            race_id = race.raceid
            for dog in race.dogs:
                dog.gru_cell = dog.gru_cell.to('cuda:0')

    def to_cpu(self):
        for race in self.racesDict.values():
            race_id = race.raceid
            # race.full_input = race.full_input.to('cpu')

            for dog in race.dogs:
                dog.gru_cell = dog.gru_cell.to('cpu')
                dog.stats = dog.stats.to('cpu')

    def race_prices_to_prob(self):
        for r in self.racesDict.values():
            # 
            # r.implied_prob = torch.tensor([(1/(torch.tensor(x)+0.0001))/(torch.tensor(x).sum()+0.0001) for x in r.start_prices],device=self.device)
            # if sum()
            if sum(r.start_prices)==0:
                r.implied_prob = torch.tensor([(1/(torch.tensor(x)+0.0001))/(torch.tensor(x).sum()+0.0001) for x in r.start_prices],device=self.device)
                r.prob = torch.tensor((1/(torch.tensor(r.start_prices)+0.0001))/((1/torch.tensor(r.start_prices)).sum()+0.0001),device=self.device)
            else:
                start_prices = [x if x>1 else 100 for x in r.start_prices]
                # print(start_prices)

                r.prob = torch.tensor(goto_conversion(start_prices),device=self.device)
                r.implied_prob = torch.tensor(goto_conversion(start_prices),device=self.device)

    def create_new_weights(self):
        races = self.racesDict.values()
        tracks = pd.Series([x.track_name for x in races])
        tracks.unique()

        weights = {}

        for t in tracks.unique():
            weights[t] = []
        for r in races:
            weights[r.track_name].append(r.classes)

        for k,v in weights.items():
            stacked = torch.stack(v)
            box_sum = stacked.sum(0)
            tot_sum = stacked.sum()
            new_weight = 1-box_sum/tot_sum
            weights[k] = new_weight

        for r in races:
            win_class = torch.argmax(r.classes)
            r.new_weights = weights[r.track_name]
            r.new_win_weight = weights[r.track_name][win_class-1]

    def create_new_weights_v2(self):
        races = self.racesDict.values()
        tracks = pd.Series([(x.track_name,x.race_dist.item()) for x in races])
        print(tracks)
        tracks.unique()

        weights = {}

        for t in tracks.unique():
            weights[t] = []
        for r in races:
            weights[(r.track_name,r.race_dist.item())].append(r.classes)

        for k,v in weights.items():
            stacked = torch.stack(v)
            box_sum = stacked.sum(0)
            tot_sum = stacked.sum()
            new_weight = 1-box_sum/tot_sum
            weights[k] = new_weight

        for r in races:
            win_class = torch.argmax(r.classes)
            r.new_weights = weights[(r.track_name,r.race_dist.item())]
            r.new_win_weight = weights[(r.track_name,r.race_dist.item())][win_class-1]

    def adjust_weights(self, weight_adj):
        races = self.racesDict.values()
        tracks = pd.Series([x.track_name for x in races])
        tracks.unique()

        for r in tqdm(races):
            if r.track_name in weight_adj.keys():
                r.new_win_weight = r.new_win_weight*weight_adj[r.track_name]
                r.new_weights = r.new_weights*weight_adj[r.track_name]

    def get_dog_train(self, idx):
        dog_idx = operator.itemgetter(*idx)
        dog_batch_id = dog_idx(self.train_dog_ids)
        races = [self.train_dogs[x].train for x in dog_batch_id] 
        return races 
        
    def get_dog_test(self, idx):
        dog_idx = operator.itemgetter(*idx)
        dog_batch_id = dog_idx(self.test_dog_ids)
        races = [self.test_dogs[x].test for x in dog_batch_id] 
        return races 

    def get_dog_val(self, idx):
        dog_idx = operator.itemgetter(*idx)
        dog_batch_id = dog_idx(self.val_dog_ids)
        races = [self.val_dogs[x].val for x in dog_batch_id] 
        return races 

    def attach_races_to_dog_input(self):
        for k,v in tqdm(self.dogsDict.items()):
            for race_id,dog_input in v.races.items():
                dog_input.race = self.racesDict[race_id]
                dog_input.full_input = torch.cat([dog_input.stats,dog_input.race.full_input], dim = 0)

    def attach_races_to_dog_inputv2(self):
        for k,v in tqdm(self.dogsDict.items()):
            for race_id,dog_input in v.races.items():
                dog_input.race = self.racesDict[race_id]
                dog_input.full_input = torch.cat([dog_input.race.race_dist,dog_input.race.race_track,dog_input.stats], dim = 0)
        
    def create_batches(self,end_date="2023-06-01", batch_days = 365, stat_mask=None):
        start_date = datetime.datetime.strptime("2019-12-01", "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()-datetime.timedelta(1)
        period = start_date
        batches = []
        while start_date<end_date:
            if end_date-start_date<2*datetime.timedelta(batch_days):
                period = end_date
            period = min(start_date+datetime.timedelta(batch_days), end_date)
            batches.append((start_date,period))
            start_date=period
        print(batches)

        batch_races_ids = [] # list of race_ids
        j = 0
        current_batch = []
        for i,r in enumerate(self.raceIDs):   
            _,end_date = batches[j]
            race_date = self.racesDict[r].race_date
            if race_date>end_date:
                print(end_date)
                if current_batch:
                    batch_races_ids.append(current_batch)
                current_batch = []
                j += 1
                if j>len(batches)-1:
                    break
            else:
                current_batch.append(r)
        print(f"Train examples {[len(x) for x in batch_races_ids]}")

        train_dogs = []
        train_dog_input = []
        for bi, batch in enumerate(tqdm(batch_races_ids)):
            batch_dogs = []
            batch_dog_input = []
            for i in tqdm(self.dog_ids):
                dog = self.dogsDict[i]
                train = [dog.races[x] for x in batch if x in dog.races.keys()]
                if train:
                    batch_dogs.append(dog)
                    batch_dog_input.append(train)
            if batch_dogs:
                train_dogs.append(batch_dogs)
                train_dog_input.append(batch_dog_input)

        
        batch_races = [[self.racesDict[r] for r in inner] for inner in batch_races_ids]

        print(f"Train examples {[len(x) for x in train_dogs]}")
        print(f"Train examples {[len(x) for x in train_dog_input]}")
        print(f"Train examples {[len(x) for x in batch_races]}")
        print(f"Train examples {[len(x) for x in batch_races_ids]}")

        test_idx = range(0,len(self.test_dog_ids))
        val_idx = range(0,len(self.val_dog_ids))
        packed_x = ""#[pack_sequence([torch.stack(n,0) for n in [[z.full_input for z in inner] for inner in x]], enforce_sorted=False).to('cuda:0') for x in train_dog_input]
        packed_y = ""#pack_sequence([torch.stack(n,0) for n in [[z.full_input.to('cuda:0') for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]], enforce_sorted=False)
        if stat_mask != None:
            packed_x_basic =[pack_sequence([torch.stack(n,0)for n in [[z.stats.masked_select(stat_mask)for z in inner] for inner in x]], enforce_sorted=False) for x in train_dog_input if x]
            packed_y_basic = pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(stat_mask) for z in inner] for inner in [x for x in self.get_dog_test(test_idx)]]], enforce_sorted=False)
            packed_v_basic = pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(stat_mask) for z in inner] for inner in [x for x in self.get_dog_val(val_idx)]]], enforce_sorted=False)
        else:
            packed_x_basic = [pack_sequence([torch.stack(n,0)for n in [[z.stats for z in inner] for inner in x]], enforce_sorted=False) for x in train_dog_input if x]
            packed_y_basic = pack_sequence([torch.stack(n,0) for n in [[z.stats for z in inner] for inner in [x for x in self.get_dog_test(test_idx)]]], enforce_sorted=False)
            packed_v_basic = pack_sequence([torch.stack(n,0) for n in [[z.stats for z in inner] for inner in [x for x in self.get_dog_val(val_idx)]]], enforce_sorted=False)           

        self.batches = {'num_batches':len(train_dogs),
                  'dogs':train_dogs,
                  'train_dog_input':train_dog_input,
                  'batch_races':batch_races,
                  'batch_races_ids':batch_races_ids,
                  'packed_x':packed_x,
                  'packed_x_basic':packed_x_basic,
                  'packed_y_basic':packed_y_basic,
                  'packed_v_basic':packed_v_basic,
                  'packed_y':packed_y}

    def margin_from_dog_to_race(self):
        for r in self.test_race_ids:
            race = self.racesDict[r]
            outputs = []
            for d in race.dogs:
                outputs.append(d.output)
            race.margins = torch.cat(outputs)
            # race.output = F.softmin(torch.stack(outputs))

    def margin_from_dog_to_race_v2(self, mode='train'):
        if mode=='train':
            for r in self.train_race_ids:
                race = self.racesDict[r]
                outputs = []
                for d in race.dogs:
                    outputs.append(d.hidden_out)
                race.hidden_in = torch.cat(outputs)
        elif mode=='test':
            for r in self.test_race_ids:
                race = self.racesDict[r]
                outputs = []
                for d in race.dogs:
                    outputs.append(d.hidden_out)
                race.hidden_in = torch.cat(outputs)
        elif mode=='val':
            for r in self.val_race_ids:
                race = self.racesDict[r]
                outputs = []
                for d in race.dogs:
                    outputs.append(d.hidden_out)
                race.hidden_in = torch.cat(outputs)

    def margin_from_dog_to_race_v3(self, mode='train', batch_races = []):
        if batch_races:
            for r in batch_races:
                race = self.racesDict[r]
                outputs = [d.hidden_out for d in race.dogs]
                race.hidden_in = torch.cat((race.race_dist,race.race_track,torch.cat(outputs)))
                
        elif mode=='train':
            for r in self.train_race_ids:
                race = self.racesDict[r]
                outputs = []
                for d in race.dogs:
                    outputs.append(d.hidden_out)
                race.hidden_in = torch.cat((race.race_dist,race.race_track,torch.cat(outputs)))
        elif mode=="test":
            for r in self.test_race_ids:
                race = self.racesDict[r]
                outputs = []
                for d in race.dogs:
                    outputs.append(d.hidden_out)
                race.hidden_in = torch.cat((race.race_dist,race.race_track,torch.cat(outputs)))
                race.relu = torch.cat(outputs).detach()
        elif mode=="val":
            for r in self.val_race_ids:
                # print("running val")
                race = self.racesDict[r]
                outputs = []
                for d in race.dogs:
                    outputs.append(d.hidden_out)
                race.hidden_in = torch.cat((race.race_dist,race.race_track,torch.cat(outputs)))
                race.relu = torch.cat(outputs).detach()

    def reset_hidden(self, num_layers=2,hidden_size=None,device='cuda:0'):
        if hidden_size==None:
            hidden_size = self.hidden_size

        self.dogsDict['nullDog'].input.hidden_out = (-torch.ones(hidden_size)).to(device)
        # self.dogsDict['nullDog'].race_input = (-torch.ones(hidden_size)).to('cuda:0')
        for dog in self.dogsDict.values():
            dog.hidden = torch.zeros(num_layers,hidden_size).to(device)
            dog.hidden_test = torch.zeros(num_layers,hidden_size).to(device)

    def detach_hidden(self, dog_list=None):
        if dog_list==None:
            dog_list = self.dogsDict.values()
        for dog in self.dogsDict.values():
            dog.hidden = dog.hidden.detach()

class GRUNetv3(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1):
        super(GRUNetv3, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)
        
        self.rl1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear((hidden_size * 8)+70, fc0_size)
        self.rl2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc0_size, fc1_size)
        self.rl3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    # x represents our data
    def forward(self, x,h=None, p1=True, warmup=False):

        if warmup:
            x = x.float()
            
            x,hidden = self.gru(x)
            x = self.relu0(x.data)
            x = self.fc0(x)
            output = x

            return output

        if p1:
            x = x.float()
            
            x,hidden = self.gru(x, h)
            x = unpack_sequence(x)
            outs = []
            for dog in x:
                outs.append(dog)

            return outs,hidden
        else:
            x = x.float()
            x = self.rl1(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.rl2(x)
            x = self.drop2(x)
            x = self.fc2(x)
            x_rl3 = self.rl3(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)

            output = self.output_fn(x), x_rl3
            return output


class GRUNetv3_LN(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1):
        super(GRUNetv3_LN, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)

        self.layer_norm = nn.LayerNorm(input_size)

        # self.layer_norm2 = nn.LayerNorm((hidden_size * 8)+70)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear((hidden_size * 8)+70, (hidden_size * 8)+70)
        # self.drop0 = nn.Dropout(dropout)
        
        self.rl1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear((hidden_size * 8)+70, fc0_size)
        self.rl2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc0_size, fc1_size)
        self.rl3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    # x represents our data
    def forward(self, x,h=None, p1=True, warmup=False):

        if warmup:
            x = x.float()
            
            x,hidden = self.gru(x)
            x = self.relu0(x.data)
            x = self.fc0(x)
            output = x

            return output

        if p1:
            x = x.float()
            x = x._replace(data=self.layer_norm(x.data))
            

            x,hidden = self.gru(x, h)
            x = unpack_sequence(x)
            outs = []
            for dog in x:
                outs.append(dog)

            return outs,hidden
        else:
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.rl1(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.rl2(x)
            x = self.drop2(x)
            x = self.fc2(x)
            x_rl3 = self.rl3(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)

            output = self.output_fn(x), x_rl3
            return output

class GRUNetv3_BN_LN(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1):
        super(GRUNetv3_BN_LN, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)

        self.layer_norm = nn.BatchNorm1d(input_size)

        self.layer_norm2 = nn.LayerNorm((hidden_size * 8)+70)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear((hidden_size * 8)+70, (hidden_size * 8)+70)
        # self.drop0 = nn.Dropout(dropout)
        
        self.rl1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear((hidden_size * 8)+70, fc0_size)
        self.rl2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc0_size, fc1_size)
        self.rl3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    # x represents our data
    def forward(self, x,h=None, p1=True, warmup=False):

        if warmup:
            x = x.float()
            
            x,hidden = self.gru(x)
            x = self.relu0(x.data)
            x = self.fc0(x)
            output = x

            return output

        if p1:
            x = x.float()
            x = x._replace(data=self.layer_norm(x.data))
            

            x,hidden = self.gru(x, h)
            x = unpack_sequence(x)
            outs = []
            for dog in x:
                outs.append(dog)

            return outs,hidden
        else:
            x = x.float()
            x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.rl1(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.rl2(x)
            x = self.drop2(x)
            x = self.fc2(x)
            x_rl3 = self.rl3(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)

            output = self.output_fn(x), x_rl3
            return output

class GRUNetv3_BN(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1):
        super(GRUNetv3_BN, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)

        self.layer_norm = nn.BatchNorm1d(input_size)

        # self.layer_norm2 = nn.LayerNorm((hidden_size * 8)+70)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear((hidden_size * 8)+70, (hidden_size * 8)+70)
        # self.drop0 = nn.Dropout(dropout)
        
        self.rl1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear((hidden_size * 8)+70, fc0_size)
        self.rl2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc0_size, fc1_size)
        self.rl3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    # x represents our data
    def forward(self, x,h=None, p1=True, warmup=False):

        if warmup:
            x = x.float()
            
            x,hidden = self.gru(x)
            x = self.relu0(x.data)
            x = self.fc0(x)
            output = x

            return output

        if p1:
            x = x.float()
            x = x._replace(data=self.layer_norm(x.data))
            

            x,hidden = self.gru(x, h)
            x = unpack_sequence(x)
            outs = []
            for dog in x:
                outs.append(dog)

            return outs,hidden
        else:
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.rl1(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.rl2(x)
            x = self.drop2(x)
            x = self.fc2(x)
            x_rl3 = self.rl3(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)

            output = self.output_fn(x), x_rl3
            return output

class GRUNetv3_BN(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1):
        super(GRUNetv3_BN, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)

        self.layer_norm = nn.BatchNorm1d(input_size)

        # self.layer_norm2 = nn.LayerNorm((hidden_size * 8)+70)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear((hidden_size * 8)+70, (hidden_size * 8)+70)
        # self.drop0 = nn.Dropout(dropout)
        
        self.rl1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear((hidden_size * 8)+70, fc0_size)
        self.rl2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc0_size, fc1_size)
        self.rl3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    # x represents our data
    def forward(self, x,h=None, p1=True, warmup=False):

        if warmup:
            x = x.float()
            
            x,hidden = self.gru(x)
            x = self.relu0(x.data)
            x = self.fc0(x)
            output = x

            return output

        if p1:
            x = x.float()
            x = x._replace(data=self.layer_norm(x.data))
            

            x,hidden = self.gru(x, h)
            x = unpack_sequence(x)
            outs = []
            for dog in x:
                outs.append(dog)

            return outs,hidden
        else:
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.rl1(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.rl2(x)
            x = self.drop2(x)
            x = self.fc2(x)
            x_rl3 = self.rl3(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)

            output = self.output_fn(x), x_rl3
            return output
        
class GRUNetv3_BN_double(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1):
        super(GRUNetv3_BN_double, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)

        self.layer_norm = nn.BatchNorm1d(input_size)

        # self.layer_norm2 = nn.LayerNorm((hidden_size * 8)+70)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear((hidden_size * 8)+70, (hidden_size * 8)+70)
        # self.drop0 = nn.Dropout(dropout)
        
        self.rl1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear((hidden_size * 8)+70, fc0_size)
        self.rl2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        #regular
        self.fc2 = nn.Linear(fc0_size, fc1_size)
        self.rl3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        #price
        self.price_fc2 = nn.Linear(fc0_size, fc1_size)
        self.price_rl3 = nn.ReLU()
        self.price_drop3 = nn.Dropout(dropout)
        self.price_fc3 = nn.Linear(fc1_size, 8)


        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    # x represents our data
    def forward(self, x,h=None, p1=True, warmup=False):

        if warmup:
            x = x.float()
            
            x,hidden = self.gru(x)
            x = self.relu0(x.data)
            x = self.fc0(x)
            output = x

            return output

        if p1:
            x = x.float()
            x = x._replace(data=self.layer_norm(x.data))
            

            x,hidden = self.gru(x, h)
            x = unpack_sequence(x)
            outs = []
            for dog in x:
                outs.append(dog)

            return outs,hidden
        else:
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.rl1(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.rl2(x)
            x_e = self.drop2(x)
            #regular
            x = self.fc2(x_e)
            x_rl3 = self.rl3(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)
            #price
            x_p = self.fc2(x_e)
            x_p_rl3 = self.rl3(x_p)
            x_p = self.drop3(x_p_rl3)
            x_p = self.fc3(x_p)

            output = self.output_fn(x), x_rl3, self.output_fn(x_p),
            return output
        
class GRUNetv3_BN_triple(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1):
        super(GRUNetv3_BN_double, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)

        self.layer_norm = nn.BatchNorm1d(input_size)

        # self.layer_norm2 = nn.LayerNorm((hidden_size * 8)+70)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear((hidden_size * 8)+70, (hidden_size * 8)+70)
        # self.drop0 = nn.Dropout(dropout)
        
        self.rl1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear((hidden_size * 8)+70, fc0_size)
        self.rl2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        #regular
        self.fc2 = nn.Linear(fc0_size, fc1_size)
        self.rl3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        #price
        self.price_fc2 = nn.Linear(fc0_size, fc1_size)
        self.price_rl3 = nn.ReLU()
        self.price_drop3 = nn.Dropout(dropout)
        self.price_fc3 = nn.Linear(fc1_size, 8)


        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    # x represents our data
    def forward(self, x,h=None, p1=True, warmup=False):

        if warmup:
            x = x.float()
            
            x,hidden = self.gru(x)
            x = self.relu0(x.data)
            x = self.fc0(x)
            output = x

            return output

        if p1:
            x = x.float()
            x = x._replace(data=self.layer_norm(x.data))
            

            x,hidden = self.gru(x, h)
            x = unpack_sequence(x)
            # outs = []
            # for dog in x:
                # outs.append(dog)

            return x,hidden
        else:
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.rl1(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.rl2(x)
            x_e = self.drop2(x)
            #regular
            x = self.fc2(x_e)
            x_rl3 = self.rl3(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)
            #price
            x_p = self.fc2(x_e)
            x_p_rl3 = self.rl3(x_p)
            x_p = self.drop3(x_p_rl3)
            x_p = self.fc3(x_p)

            output = self.output_fn(x), x_rl3, self.output_fn(x_p), # changed
            return output

class AttnNetv1(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1):
        super(AttnNetv1, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(input_size,input_size/2, dropout=0.1)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)
        
        self.rl1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear((input_size * 8)+70, fc0_size)
        self.rl2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc0_size, fc1_size)
        self.rl3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    # x represents our data
    def forward(self, x,h=None, p1=True, warmup=False):


        if p1:
            lens = x[1]
            x = x[0].float()
            print(x[0].shape)
            x = self.encoder(x)
            x = unpad_sequence(x, lens)
            outs = []
            for dog in x:
                outs.append(dog)

            return outs
        else:
            x = x.float()
            x = self.rl1(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.rl2(x)
            x = self.drop2(x)
            #regular
            x = self.fc2(x)
            x_rl3 = self.rl3(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)
            #price
            x_p = self.fc2(x)
            x_p_rl3 = self.rl3(x_p)
            x_p = self.drop3(x_p_rl3)
            x_p = self.fc3(x_p)

            output = self.output_fn(x), x_rl3, self.output_fn(x_p),
            return output