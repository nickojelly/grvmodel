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
import wandb
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict

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

        self.cell_out = None
        self.box = dog_box
        self.gru_filled = 0
        self.margin = margin
        # self.output = torch.tensor([100.1]).to('cuda:0')

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
        self.races :dict[str,DogInput] = {} #
        self.race_train = [] #[DogInput]
        self.races_test = [] #[DogInput]

    def add_races(self, raceid, racedate, stats,nextraceid, prevraceid, box,margin=None, bfsp=None, sp=None):
        self.races[raceid] = DogInput(self.dogid, raceid, stats, self, box, torch.ones(self.hidden_size), bfsp, sp, margin, self.hidden_size) #this is the change
        self.races[raceid].nextrace(nextraceid)
        self.races[raceid].prevrace(prevraceid)

class Race:
    def __init__(self, raceid,trackOHE, dist, state,classes=None):
        self.raceid = raceid
        self.race_dist = dist.to('cuda:0')
        self.race_track = trackOHE.to('cuda:0')
        self.track_name = None
        self.raw_margins = None
        self.raw_places = None
        self.raw_prices = None
        self.state = state
        self.hidden_in:torch.Tensor = None

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
        self.dogs:list[DogInput] = dogs_list

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

    def list_dog_boxes(self):
        dogs_l = [x.box for x in self.dogs]
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
        self.racesDict :dict[str,Race] = {}
        self.dogsDict :dict[str,Dog] = {}
        self.raceIDs = []
        self.dog_ids = []
        self.hidden_size = hidden_size
        self.layers = layers
        self.batches_setup = False
        self.getter = operator.itemgetter(*range(batch_size))
        self.device = device

    def add_race(self,raceid:str, trackOHE, dist,state, classes=None):
        self.racesDict[raceid] = Race(raceid, trackOHE, dist,state, classes)
        # self.raceIDs.append(raceid)

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
            if race_date>=val_race_date:
                self.val_race_ids.append(r)
                val += 1
            elif race_date>=start_date:
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
        races = [self.racesDict[x] for x in self.test_race_ids] #Returns list of class Race
        return races #self.racesDict[raceidx]

    def get_val_input(self, idx, Train=True) -> Race:
        races = [self.racesDict[x] for x in self.val_race_ids] #Returns list of class Race
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

    def create_hidden_in_dict(self):
        self.hidden_ins_dict = {}
        self.output_dict = {}
        for race_id,race in self.racesDict.items():
            hidden_in = race.hidden_in
            self.hidden_ins_dict[race_id] = race.hidden_in.detach()
            self.output_dict[race_id] = race.output

    def load_hidden_in_dict(self,hidden_in_dict,output_dict):
        missing = []
        for race_id,race in self.racesDict.items():
            try:
                race.hidden_in = hidden_in_dict[race_id]
                race.output = output_dict[race_id]
            except Exception as e:
                print(f"{race_id} not in dict")
                missing.append(race_id)
        
        print(f"Missing {len(missing)} out of {len(self.racesDict)}")
        print(f"Dropping {len(missing)}")
        # for m in missing:
        #     self.racesDict.pop(m)

        # self.raceIDs = [r for r in self.raceIDs if r not in missing]
        


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
            r.prices_tensor = torch.tensor(r.start_prices,device='cuda:0')
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

    def create_batches(self,end_date="2023-06-01", batch_days = 365, stat_mask=None,data_mask=None,gen_packed_seq=True):
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
                current_batch.append(r)
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
        for bi, batch in tqdm(enumerate(batch_races_ids)):
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
        output_batch = None
        output_p_batch = None
        if gen_packed_seq:
            if stat_mask != None:
                packed_x_basic =[pack_sequence([torch.stack(n,0)for n in [[z.stats.masked_select(stat_mask)for z in inner] for inner in x]], enforce_sorted=False) for x in train_dog_input if x]
                packed_y_basic = pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(stat_mask) for z in inner] for inner in [x for x in self.get_dog_test(test_idx)]]], enforce_sorted=False)
                packed_v_basic = pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(stat_mask) for z in inner] for inner in [x for x in self.get_dog_val(val_idx)]]], enforce_sorted=False)
                self.packed_x_data =[pack_sequence([torch.stack(n,0)for n in [[z.stats.masked_select(data_mask)for z in inner] for inner in x]], enforce_sorted=False) for x in train_dog_input if x]
                self.packed_y_data = pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(data_mask) for z in inner] for inner in [x for x in self.get_dog_test(test_idx)]]], enforce_sorted=False)
                self.packed_v_data = pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(data_mask) for z in inner] for inner in [x for x in self.get_dog_val(val_idx)]]], enforce_sorted=False)
            else:
                packed_x_basic = [pack_sequence([torch.stack(n,0)for n in [[z.stats for z in inner] for inner in x]], enforce_sorted=False) for x in train_dog_input if x]
                packed_y_basic = pack_sequence([torch.stack(n,0) for n in [[z.stats for z in inner] for inner in [x for x in self.get_dog_test(test_idx)]]], enforce_sorted=False)
                packed_v_basic = pack_sequence([torch.stack(n,0) for n in [[z.stats for z in inner] for inner in [x for x in self.get_dog_val(val_idx)]]], enforce_sorted=False)
        else: 
            packed_x_basic = None
            packed_y_basic = None
            packed_v_basic = None
            output_batch = [torch.stack([r.output[0] for r in batch]) for batch in batch_races]
            output_p_batch = [torch.stack([r.output[1] for r in batch]) for batch in batch_races]

        self.batches = {'num_batches':len(train_dogs),
                'dogs':train_dogs,
                'train_dog_input':train_dog_input,
                'batch_races':batch_races,
                'batch_races_ids':batch_races_ids,
                'packed_x':packed_x,
                'packed_x_basic':packed_x_basic,
                'packed_y_basic':packed_y_basic,
                'packed_v_basic':packed_v_basic,
                'packed_y':packed_y,
                'output_batch':output_batch,
                'output_p_batch':output_p_batch
                }
        
    def create_batches_w_states(self,end_date="2023-06-01", batch_days = 365, stat_mask=None,data_mask=None):
        start_date = datetime.datetime.strptime("2019-12-01", "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()-datetime.timedelta(1)
        period = start_date
        batches = []
        # Generate date ranges, 
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
                current_batch.append(r)
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
            packed_x_basic = [pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(stat_mask) for z in inner] for inner in x]], enforce_sorted=False) for x in train_dog_input if x]
            packed_y_basic = pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(stat_mask) for z in inner] for inner in [x for x in self.get_dog_test(test_idx)]]], enforce_sorted=False)
            packed_v_basic = pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(stat_mask) for z in inner] for inner in [x for x in self.get_dog_val(val_idx)]]], enforce_sorted=False)
            self.packed_x_data =[pack_sequence([torch.stack(n,0)for n in [[z.stats.masked_select(data_mask)for z in inner] for inner in x]], enforce_sorted=False) for x in train_dog_input if x]
            self.packed_y_data = pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(data_mask) for z in inner] for inner in [x for x in self.get_dog_test(test_idx)]]], enforce_sorted=False)
            self.packed_v_data = pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(data_mask) for z in inner] for inner in [x for x in self.get_dog_val(val_idx)]]], enforce_sorted=False)
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
        states = self.states 
        # for batch in batch_races:
        #    {} 


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

    def reset_hidden_w_param(self, hidden_state_param,num_layers=2,hidden_size=None,device='cuda:0'):
        if hidden_size==None:
            hidden_size = self.hidden_size

        self.dogsDict['nullDog'].input.hidden_out = (-torch.ones(hidden_size)).to(device)
        # self.dogsDict['nullDog'].race_input = (-torch.ones(hidden_size)).to('cuda:0')
        for dog in self.dogsDict.values():
            dog.hidden = hidden_state_param
            dog.cell = hidden_state_param
            dog.hidden_test = hidden_state_param
            dog.cell_test = hidden_state_param

    def reset_hidden(self, num_layers=2,hidden_size=None,device='cuda:0'):
        if hidden_size==None:
            hidden_size = self.hidden_size

        self.dogsDict['nullDog'].input.hidden_out = (-torch.ones(hidden_size)).to(device)
        # self.dogsDict['nullDog'].race_input = (-torch.ones(hidden_size)).to('cuda:0')
        for dog in self.dogsDict.values():
            dog.hidden = torch.zeros(num_layers,hidden_size).to(device)
            dog.cell = torch.zeros(num_layers,hidden_size).to(device)
            dog.hidden_test = torch.zeros(num_layers,hidden_size).to(device)
            dog.cell_test = torch.zeros(num_layers,hidden_size).to(device)

    def detach_hidden(self, dog_list=None):
        if dog_list==None:
            dog_list = self.dogsDict.values()
        for dog in self.dogsDict.values():
            dog.hidden = dog.hidden.detach()
            dog.cell = dog.cell.detach()

    def del_hidden(self):
        for dog in self.dogsDict.values():
            del dog.hidden
            del dog.cell
            for dog_input in dog.races.values():
                try:
                    del dog_input.hidden_out
                except:
                    pass

    def my_collate(self,batch):
        data = [self.racesDict[x] for x in batch]
        return data


    def data_loader(self, batch_size, mode='train', shuffle=True):
        self.dataset = self.train_race_ids
        train_sampler = SubsetRandomSampler(list(range(len(self.dataset))))
                                            
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,  collate_fn=self.my_collate, shuffle=shuffle)
        
        return train_loader
    

    def data_loader_extra_shuffle(self, batch_size, mode='train', shuffle=True):
        self.dataset = self.train_race_ids
        train_sampler = SubsetRandomSampler(list(range(len(self.dataset))))
                                            
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,  collate_fn=self.my_collate, shuffle=shuffle)
        
        return train_loader

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
        
class Attention_w_viz(nn.Module):
    def __init__(self, hidden_size):
        super(Attention_w_viz, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, viz=False):
        # Compute attention scores
        print(hidden_states.shape)
        attention_scores = self.fc(hidden_states)

        # Create mask to exclude future states
        seq_length = hidden_states.size(0)
        future_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().cuda()
        attention_scores = attention_scores.repeat(1, 1, seq_length)
        future_mask = future_mask.unsqueeze(1).repeat(1,attention_scores.shape[1], 1)
        attention_scores.masked_fill_(future_mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=0)
        indices = torch.arange(seq_length).to(attention_weights.device)
        selected_attention_weights = attention_weights[indices, :, indices].unsqueeze(-1)
        print(selected_attention_weights.squeeze()[:, 0:20])

        if viz:
            attention_weights_np = selected_attention_weights.detach().squeeze().cpu().numpy()

            attention_weights_np = attention_weights_np[:, 0:20]

            print(attention_weights_np.shape)

            # Transpose array if necessary
            attention_weights_np = attention_weights_np

            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(attention_weights_np, cmap='hot', interpolation='nearest')
            fig.colorbar(im, ax=ax, label='Attention weight')
            ax.set_title('Attention Weights')
            plt.show()
            wandb.log({"attention_weights": wandb.Image(fig)})

        return selected_attention_weights

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, viz=False):
        # Compute attention scores
        print(hidden_states.shape)
        attention_scores = self.fc(hidden_states)
        attention_scores = torch.sum(attention_scores, dim=2)

        # Create mask to exclude future states
        seq_length = hidden_states.size(0)
        future_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().cuda()
        attention_scores = attention_scores.repeat(1, 1, seq_length)
        future_mask = future_mask.unsqueeze(1).repeat(1,attention_scores.shape[1], 1)
        attention_scores.masked_fill_(future_mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=0)
        indices = torch.arange(seq_length).to(attention_weights.device)
        selected_attention_weights = attention_weights[indices, :, indices].unsqueeze(-1)


        return selected_attention_weights

class Attention_simple(nn.Module):
    def __init__(self, hidden_size):
        super(Attention_simple, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, viz=False):
        # Compute attention scores
        # print(hidden_states.shape)
        attention_scores = self.fc(hidden_states)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=0)
        indices = torch.arange(hidden_states.size(0)).to(attention_weights.device)
        # print(attention_weights.shape)
        # selected_attention_weights = attention_weights[indices, :, indices].unsqueeze(-1)

        return attention_weights

class Dynamic_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Dynamic_Attention, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, viz=False):
        # print(f"{hidden_states.shape=}")
        seq_length = hidden_states.size(0)
        selected_attention_weights = []

        for i in range(seq_length):
            # Compute attention scores based on current and previous states
            attention_scores = self.fc(hidden_states[:i+1])
            # print(attention_scores.shape)
            attention_scores = torch.sum(attention_scores, dim=2,keepdim=True)

            # Create mask to exclude future states
            future_mask = torch.triu(torch.ones(i+1, i+1), diagonal=1).bool().to(hidden_states.device)
            attention_scores = attention_scores.repeat(1, 1, i+1)
            future_mask = future_mask.unsqueeze(1).repeat(1,attention_scores.shape[1], 1)
            # future_mask = future_mask.squeeze(-1)
            attention_scores.masked_fill_(future_mask, float('-inf'))

            # Apply softmax to get attention weights
            attention_weights = F.softmax(attention_scores, dim=0)

            indices = torch.arange(i+1).to(attention_weights.device)
            attention_weights = attention_weights[indices, :, indices].unsqueeze(-1)

            # print(attention_weights.shape)

            # Select the attention weight for the current state
            selected_attention_weights.append(attention_weights[-1])
            # print(attention_weights[-1].shape)

        # Stack the selected attention weights into a tensor
        selected_attention_weights = torch.stack(selected_attention_weights)

        return selected_attention_weights
    
class GRUNetv3_extra_attn(nn.Module):
    def __init__(self,input_size,hidden_size,hidden=None,output='raw',dropout=0.3,fc0_size=256,fc1_size=64,num_layers=1,data_mask_size=None):
        super(GRUNetv3_extra_attn, self).__init__()
        self.gru = nn.GRU(input_size,
                          hidden_size,
                          num_layers=num_layers,
                          dropout=0.3)
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size, 1)

        self.batch_norm = nn.BatchNorm1d(input_size)
        self.batch_norm_data = nn.BatchNorm1d(data_mask_size)
        self.attention = Dynamic_Attention(hidden_size)

        #p1
        self.fc0_p1 = nn.Linear(hidden_size+data_mask_size, hidden_size)
        self.fc0_p1_drop = nn.Dropout(dropout)
        self.fc0_p2 = nn.Linear(hidden_size, hidden_size)
        self.fc0_p2_drop = nn.Dropout(dropout)
        self.fc0_p3 = nn.Linear(hidden_size, hidden_size)
        self.fc0_p3_drop = nn.Dropout(dropout)

        #p2
        # self.layer_norm2 = nn.LayerNorm((hidden_size * 8)+70)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(((hidden_size) * 8) + 70, hidden_size * 8)
        # self.drop0 = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 8, hidden_size * 4)
        self.drop2 = nn.Dropout(dropout)

        #regular
        self.fc2 = nn.Linear(hidden_size * 4, fc1_size)
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        #price
        self.price_fc2 = nn.Linear(hidden_size * 4, fc1_size)
        self.price_drop3 = nn.Dropout(dropout)
        self.price_fc3 = nn.Linear(fc1_size, 8)

        if output == 'raw':
            self.output_fn = nn.Identity()
        elif output == 'softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output == 'log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    # x represents our data
    def forward(self, x, h=None, p1=True, warmup=False,viz=False):

        if p1:
            x, x_d = x   
            # x = self.batch_norm(x)
            x_d = x_d.float()
            x_d = x_d._replace(data=self.batch_norm_data(x_d.data))
            x = x.float()
            x_d = x_d.float()
            x = x._replace(data=self.batch_norm(x.data))

            x, hidden = self.gru(x, h)
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)  # unpack sequence
            attention_weights = self.attention(x,viz)  # apply attention
            x = attention_weights * x  # multiply attention weights with hidden states
            x = torch.nn.utils.rnn.unpad_sequence(x, lengths)  # pack sequence again
            x_d = unpack_sequence(x_d)
            # print(x.shape)
            outs = []
            
            for i, dog in enumerate(x):
                dog = torch.cat((dog, x_d[i]), dim=1)
                # dog = self.relu(dog)
                dog = self.fc0_p1(dog)
                dog = self.relu(dog)
                dog = self.fc0_p1_drop(dog)
                dog = self.fc0_p2(dog)
                dog = self.relu(dog)
                dog = self.fc0_p2_drop(dog)
                dog = self.fc0_p3(dog)
                dog = self.relu(dog)
                dog = self.fc0_p3_drop(dog)
                outs.append(dog)
            print(f"{hidden.shape=} ")
            return outs, hidden
        else:
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.relu(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.relu(x)
            x_e = self.drop2(x)
            #regular
            x = self.fc2(x_e)
            x_rl3 = self.relu(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)
            #price
            x_p = self.fc2(x_e)
            x_p_rl3 = self.relu(x_p)
            x_p = self.drop3(x_p_rl3)
            x_p = self.fc3(x_p)

            output = self.output_fn(x), x_rl3, self.output_fn(x_p),
            return output

class GRUNetv4_extra_attn(nn.Module):
    def __init__(self,input_size,hidden_size,hidden=None,output='raw',dropout=0.3,fc0_size=256,fc1_size=64,num_layers=1,data_mask_size=None):
        super(GRUNetv4_extra_attn, self).__init__()
        self.gru = nn.GRU(input_size,
                          hidden_size,
                          num_layers=num_layers,
                          dropout=0.3)
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size, 1)

        self.batch_norm = nn.BatchNorm1d(input_size)
        self.batch_norm_data = nn.BatchNorm1d(data_mask_size)
        self.attention = Dynamic_Attention(hidden_size)
        self.h0 = nn.Parameter(torch.zeros(num_layers, hidden_size))

        #p1
        self.fc0_p1 = nn.Linear(hidden_size, hidden_size)
        self.fc0_p1_drop = nn.Dropout(dropout)
        self.fc0_p2 = nn.Linear(hidden_size, hidden_size)
        self.fc0_p2_drop = nn.Dropout(dropout)
        self.fc0_p3 = nn.Linear(hidden_size, hidden_size)
        self.fc0_p3_drop = nn.Dropout(dropout)

        #p2
        # self.layer_norm2 = nn.LayerNorm((hidden_size * 8)+70)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(((hidden_size) * 8) + 70, hidden_size * 8)
        # self.drop0 = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 8, hidden_size * 4)
        self.drop2 = nn.Dropout(dropout)

        #regular
        self.fc2 = nn.Linear(hidden_size * 4, fc1_size)
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        #price
        self.price_fc2 = nn.Linear(hidden_size * 4, fc1_size)
        self.price_drop3 = nn.Dropout(dropout)
        self.price_fc3 = nn.Linear(fc1_size, 8)

        if output == 'raw':
            self.output_fn = nn.Identity()
        elif output == 'softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output == 'log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    # x represents our data
    def forward(self, x, h=None, p1=True, warmup=False,viz=False):

        if p1:
            x, x_d = x   
            # x = self.batch_norm(x)
            x_d = x_d.float()
            x_d = x_d._replace(data=self.batch_norm_data(x_d.data))
            x = x.float()
            x_d = x_d.float()
            x = x._replace(data=self.batch_norm(x.data))
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)  # unpack sequence 
            # print(f"{x.shape=} {h.shape=} {lengths=}")  
            outputs = []
            hidden_states = []

            hidden = h.contiguous()
            for t in range(x.size(0)):
                # Process one time step with GRU
                output, hidden = self.gru(x[t:t+1], hidden)
                # Compute attention scores based on all previous hidden states
                if outputs:
                    attention_weights = self.attention(torch.cat(outputs,dim=0))
                else:
                    attention_weights = self.attention(output)
                # Apply attention weights to output
                output = attention_weights[-1] * output
                outputs.append(output)
                # hidden_states.append(hidden.detach())
            outputs = torch.cat(outputs,dim=0)
            # hidden_states = torch.stack(hidden_states,dim=0)

            x = outputs
             
            x = torch.nn.utils.rnn.unpad_sequence(x, lengths)  # pack sequence again
            # hidden_states = torch.nn.utils.rnn.unpad_sequence(hidden_states, lengths)  # pack sequence again
            # hidden_states = torch.stack([x[-1] for x in hidden_states])
            # hidden = hidden_states

            x_d = unpack_sequence(x_d)
            outs = []
            for i, dog in enumerate(x):
                # dog = torch.cat((dog, x_d[i]), dim=1)
                dog = self.fc0_p1(dog)
                dog = self.relu(dog)
                dog = self.fc0_p1_drop(dog)
                dog = self.fc0_p2(dog)
                dog = self.relu(dog)
                dog = self.fc0_p2_drop(dog)
                dog = self.fc0_p3(dog)
                dog = self.relu(dog)
                dog = self.fc0_p3_drop(dog)
                outs.append(dog)

            return outs, hidden
        
        else:
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.relu(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.relu(x)
            x_e = self.drop2(x)
            #regular
            x = self.fc2(x_e)
            x_rl3 = self.relu(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)
            #price
            x_p = self.fc2(x_e)
            x_p_rl3 = self.relu(x_p)
            x_p = self.drop3(x_p_rl3)
            x_p = self.fc3(x_p)

            output = self.output_fn(x), x_rl3, self.output_fn(x_p),
            return output

class AdvNet_stacking(nn.Module):
    def __init__(self, input_size,num_models=10,dropout=0.3, fc0_size=256,fc1_size=64,data_mask_size=None):
        super(AdvNet_stacking, self).__init__()
        self.num_models = num_models
        for i in range(num_models):
            setattr(self, f'model_{i}', AdvNet(input_size,dropout,fc0_size,fc1_size,data_mask_size))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_0 = nn.Linear(8*num_models,8*num_models)
        self.fc_1 = nn.Linear(8*num_models,8)
        self.model_list = [getattr(self, f'model_{i}') for i in range(num_models)]

    def forward(self, x,p1=True):
        if p1:
            outputs = []
            for i in range(self.num_models):
                model = getattr(self, f'model_{i}')
                outputs.append(model(x[i]))

            outputs = torch.stack(outputs)
            return outputs
        else:
            outputs = []
            for i in range(self.num_models):
                model = getattr(self, f'model_{i}')
                # print(f"P2 {x.shape=}")
                outputs.append(model(x))

            outputs = torch.cat(outputs,dim=-1)
            outputs = self.fc_0(outputs)
            outputs = self.relu(outputs)
            outputs = self.dropout(outputs)
            outputs = self.fc_1(outputs)

            return outputs

class AdvNet(nn.Module):
    def __init__(self, input_size,dropout=0.3, fc0_size=256,fc1_size=64,data_mask_size=None):
        super(AdvNet, self).__init__()

        self.batch_norm = nn.LazyBatchNorm1d()
        self.dog_batch_norm = nn.LazyBatchNorm1d()  
        self.relu = nn.ReLU()

        #p1
        self.fc1_p1 = nn.LazyLinear(fc1_size)
        self.fc1_p1_drop = nn.Dropout(dropout)
        self.fc1_p2 = nn.Linear(fc0_size,fc1_size)
        self.fc1_p2_drop = nn.Dropout(dropout)
        self.fc1_p3 = nn.Linear(fc1_size,fc1_size)
        self.fc1_p3_drop = nn.Dropout(dropout)



        #p2
        self.fc0_p1 = nn.LazyLinear(fc0_size)
        self.fc0_p1_drop = nn.Dropout(dropout)
        self.fc0_p2 = nn.Linear(fc0_size,fc1_size)
        self.fc0_p2_drop = nn.Dropout(dropout)
        self.fc0_p3 = nn.Linear(fc1_size,fc1_size)
        self.fc0_p3_drop = nn.Dropout(dropout)
        self.fc0_p4 = nn.Linear(fc1_size,8)



    def forward(self, x):
        # x = torch.stack(x,dim=0)
        # print(x.shape)
        x = x.float()
        dogs = []
        x = x.transpose(0,1)
        
        for dog in x:
            # print(dog.shape)
            x = dog.float()
            x = self.dog_batch_norm(x)
            x = self.fc1_p1(x)
            x = self.relu(x)
            x = self.fc1_p1_drop(x)
            dogs.append(x)
        # print(f"{dogs[0].shape=}")
        x = torch.cat(dogs,dim=-1)
        # print(f"{x.shape=}")


        x = x.float()
        # x = self.batch_norm(x)
        x = self.fc0_p1(x)
        x = self.relu(x)
        x = self.fc0_p1_drop(x)
        x = self.fc0_p2(x)
        x = self.relu(x)
        x = self.fc0_p2_drop(x)
        x = self.fc0_p3(x)
        x = self.relu(x)
        x = self.fc0_p3_drop(x)
        x = self.fc0_p4(x)
        # print(f"p1 {x.shape=}")

        return x

        # x represents our data

class SimpleNet_stacking(nn.Module):
    def __init__(self, input_size,num_models=10,dropout=0.3, fc0_size=256,fc1_size=64,data_mask_size=None):
        super(SimpleNet_stacking, self).__init__()
        self.num_models = num_models
        for i in range(num_models):
            setattr(self, f'model_{i}', SimpleNet(input_size,dropout,fc0_size,fc1_size,data_mask_size))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_0 = nn.Linear(8*num_models,8*num_models)
        self.fc_1 = nn.Linear(8*num_models,8)

    def forward(self, x,p1=True):
        if p1:
            outputs = []
            for i in range(self.num_models):
                model = getattr(self, f'model_{i}')
                outputs.append(model(x[i]))

            outputs = torch.stack(outputs)
            return outputs
        else:
            outputs = []
            for i in range(self.num_models):
                model = getattr(self, f'model_{i}')
                outputs.append(model(x))

            outputs = torch.cat(outputs,dim=-1)
            outputs = self.fc_0(outputs)
            outputs = self.relu(outputs)
            outputs = self.dropout(outputs)
            outputs = self.fc_1(outputs)

            return outputs

class SimpleNet(nn.Module):
    def __init__(self, input_size,dropout=0.3, fc0_size=256,fc1_size=64,data_mask_size=None):
        super(SimpleNet, self).__init__()

        self.batch_norm = nn.LazyBatchNorm1d()
        self.relu = nn.ReLU()

        #p1
        self.fc0_p1 = nn.LazyLinear(fc0_size)
        self.fc0_p1_drop = nn.Dropout(dropout)
        self.fc0_p2 = nn.Linear(fc0_size,fc1_size)
        self.fc0_p2_drop = nn.Dropout(dropout)
        self.fc0_p3 = nn.Linear(fc1_size,fc1_size)
        self.fc0_p3_drop = nn.Dropout(dropout)
        self.fc0_p4 = nn.Linear(fc1_size,8)

    def forward(self, x):
        # x = torch.stack(x,dim=0)

        x = x.float()
        x = self.batch_norm(x)
        x = self.fc0_p1(x)
        x = self.relu(x)
        x = self.fc0_p1_drop(x)
        x = self.fc0_p2(x)
        x = self.relu(x)
        x = self.fc0_p2_drop(x)
        x = self.fc0_p3(x)
        x = self.relu(x)
        x = self.fc0_p3_drop(x)
        x = self.fc0_p4(x)

        return x

        # x represents our data

class GRUNetv3_simple_extra_data(nn.Module):
    def __init__(self, input_size,dropout=0.3, fc0_size=256,fc1_size=64,data_mask_size=None):
        super(GRUNetv3_simple_extra_data, self).__init__()

        self.batch_norm = nn.LazyBatchNorm1d()
        self.relu = nn.ReLU()

        #p1
        self.fc0_p1 = nn.LazyLinear(fc0_size)
        self.fc0_p1_drop = nn.Dropout(dropout)
        self.fc0_p2 = nn.Linear(fc0_size,fc1_size)
        self.fc0_p2_drop = nn.Dropout(dropout)
        self.fc0_p3 = nn.Linear(fc1_size,fc1_size)
        self.fc0_p3_drop = nn.Dropout(dropout)

            # x represents our data
    def forward(self, x):
        x = x.float()
        # x = self.batch_norm(x)
        x = self.fc0_p1(x)
        x = self.relu(x)
        x = self.fc0_p1_drop(x)
        x = self.fc0_p2(x)
        x = self.relu(x)
        x = self.fc0_p2_drop(x)
        x = self.fc0_p3(x)

        return x


class GRUNetv3_extra(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1,data_mask_size=None):
        super(GRUNetv3_extra, self).__init__()
        self.name = 'GRUNetv3_extra  '
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)
        self.h0 = nn.Parameter(torch.zeros(num_layers, hidden_size))

        self.batch_norm = nn.BatchNorm1d(input_size)
        self.batch_norm_data = nn.BatchNorm1d(data_mask_size)

        #extra data
        self.extra_1 = GRUNetv3_simple_extra_data(20,dropout,fc0_size,fc1_size)

        #p2
        # self.layer_norm2 = nn.LayerNorm((hidden_size * 8)+70)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(((hidden_size+1*fc1_size) * 8)+70, hidden_size * 8)
        # self.drop0 = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 8, hidden_size*4)
        self.drop2 = nn.Dropout(dropout)

        #regular
        self.fc2 = nn.Linear(hidden_size*4, fc1_size)
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        #price
        # self.price_fc2 = nn.Linear(hidden_size*4, fc1_size)
        # self.price_drop3 = nn.Dropout(dropout)
        # self.price_fc3 = nn.Linear(fc1_size, 8)


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
            x,x_d = x
            # x = self.batch_norm(x)
            x_d = x_d.float()
            x_d = x_d._replace(data=self.batch_norm_data(x_d.data))
            x = x.float()
            x_d = x_d.float()
            x = x._replace(data=self.batch_norm(x.data))
            x_og = x

            x,hidden = self.gru(x, h)
            x = unpack_sequence(x)
            x_d = unpack_sequence(x_d)
            x_og = unpack_sequence(x_og)
            outs = []
            for i,dog in enumerate(x):

                dog_simple = self.extra_1(x_d[i])
                dog = torch.cat((dog,dog_simple),dim=1)
                outs.append(dog)
            # print(f"{outs[0].shape=} ")
            return outs,hidden
        else:
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.relu(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.relu(x)
            x_e = self.drop2(x)
            #regular
            x = self.fc2(x_e)
            x_rl3 = self.relu(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)
            #price
            x_p = self.fc2(x_e)
            x_p_rl3 = self.relu(x_p)
            x_p = self.drop3(x_p_rl3)
            x_p = self.fc3(x_p)

            output = self.output_fn(x), x_rl3, self.output_fn(x_p),
            return output


# @torch.no_grad()
class GRUNetv3_extra_fast_inf(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1,data_mask_size=None):
        super(GRUNetv3_extra_fast_inf, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)
        self.h0 = nn.Parameter(torch.zeros(num_layers, hidden_size))

        self.batch_norm = nn.BatchNorm1d(input_size)
        self.batch_norm_data = nn.BatchNorm1d(data_mask_size)

        #extra data
        self.extra_1 = GRUNetv3_simple_extra_data(20,dropout,fc0_size,fc1_size)

        #p2
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(((hidden_size+1*fc1_size) * 8)+70, hidden_size * 8)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 8, hidden_size*4)
        self.drop2 = nn.Dropout(dropout)

        #regular
        self.fc2 = nn.Linear(hidden_size*4, fc1_size)
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        #price
        # self.price_fc2 = nn.Linear(hidden_size*4, fc1_size)
        # self.price_drop3 = nn.Dropout(dropout)
        # self.price_fc3 = nn.Linear(fc1_size, 8)


        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

        self.output_cache_p1 = {}
        self.output_cache_p2 = {}

    # x represents our data
    def forward(self, x,h=None, p1=True, warmup=False,batch_races=None):

        
        if p1:
            # print("model running p1")
            x,x_d = x
            # x = self.batch_norm(x)
            x_d = x_d.float()
            x_d = x_d._replace(data=self.batch_norm_data(x_d.data))
            x = x.float()
            x_d = x_d.float()
            x = x._replace(data=self.batch_norm(x.data))
            x_og = x

            x,hidden = self.gru(x, h)
            x = unpack_sequence(x)
            x_d = unpack_sequence(x_d)
            # x_og = unpack_sequence(x_og)
            outs = []
            for i,dog in enumerate(x):
                dog_simple = self.extra_1(x_d[i])
                dog = torch.cat((dog,dog_simple),dim=1)
                outs.append(dog)
            # print(f"{outs[0].shape=} ")
            output = outs,hidden
            self.output_cache_p1[batch_races[0].raceid]  = output
            return outs,hidden
        
        else:          
            # print("model running p2")
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.relu(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.relu(x)
            x_e = self.drop2(x)
            #regular
            x = self.fc2(x_e)
            x_rl3 = self.relu(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)
            #price
            x_p = self.fc2(x_e)
            x_p_rl3 = self.relu(x_p)
            x_p = self.drop3(x_p_rl3)
            x_p = self.fc3(x_p)

            output = self.output_fn(x).detach(), x_rl3.detach(), self.output_fn(x_p).detach(),
            
            # output = self.output_fn(x).detach(), x_rl3.detach(), self.output_fn(x_p).detach(),

            self.output_cache_p2[batch_races[0].raceid] = output

            return output
        
# @torch.no_grad()
class GRUNetv3_extra_fast_inf_price(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1,data_mask_size=None):
        super(GRUNetv3_extra_fast_inf_price, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)
        self.h0 = nn.Parameter(torch.zeros(num_layers, hidden_size))

        self.batch_norm = nn.BatchNorm1d(input_size)
        self.batch_norm_data = nn.BatchNorm1d(data_mask_size)

        #extra data
        self.extra_1 = GRUNetv3_simple_extra_data(20,dropout,fc0_size,fc1_size)

        #p2
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(((hidden_size+1*fc1_size) * 8)+70, hidden_size * 8)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 8, hidden_size*4)
        self.drop2 = nn.Dropout(dropout)

        #regular
        self.fc2 = nn.Linear(hidden_size*4, fc1_size)
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        #price
        # self.price_fc2 = nn.Linear(hidden_size*4, fc1_size)
        # self.price_drop3 = nn.Dropout(dropout)
        # self.price_fc3 = nn.Linear(fc1_size, 8)


        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

        self.output_cache_p1 = {}
        self.output_cache_p2 = {}

    # x represents our data
    def forward(self, x,h=None, p1=True, warmup=False,batch_races=None):

        
        if p1:
            # print("model running p1")
            x,x_d = x
            # x = self.batch_norm(x)
            x_d = x_d.float()
            x_d = x_d._replace(data=self.batch_norm_data(x_d.data))
            x = x.float()
            x_d = x_d.float()
            x = x._replace(data=self.batch_norm(x.data))
            x_og = x

            x,hidden = self.gru(x, h)
            x = unpack_sequence(x)
            x_d = unpack_sequence(x_d)
            x_og = unpack_sequence(x_og)
            outs = []
            for i,dog in enumerate(x):
                dog_simple = self.extra_1(x_d[i])
                dog = torch.cat((dog,dog_simple),dim=1)
                outs.append(dog)
            # print(f"{outs[0].shape=} ")
            output = outs,hidden
            self.output_cache_p1[batch_races[0].raceid]  = output
            return outs,hidden
        
        else:
            # print("model running p2")
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.relu(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.relu(x)
            x_e = self.drop2(x)
            #regular
            x = self.fc2(x_e)
            x_rl3 = self.relu(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)
            #price
            x_p = self.fc2(x_e)
            x_p_rl3 = self.relu(x_p)
            x_p = self.drop3(x_p_rl3)
            x_p = self.fc3(x_p)

            output = self.output_fn(x), x_rl3, self.output_fn(x_p),

            self.output_cache_p2[batch_races[0].raceid] = output

            return output


class GRUNetv3_extra_price(nn.Module):
    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1,data_mask_size=None):
        super(GRUNetv3_extra_price, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)
        self.h0 = nn.Parameter(torch.zeros(num_layers, hidden_size))

        self.batch_norm = nn.BatchNorm1d(input_size)
        self.batch_norm_data = nn.BatchNorm1d(data_mask_size)

        #extra data
        self.extra_1 = GRUNetv3_simple_extra_data(20,dropout,fc0_size,fc1_size)

        #p2
        # self.layer_norm2 = nn.LayerNorm((hidden_size * 8)+70)
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(((hidden_size+1*fc1_size) * 8)+70, hidden_size * 8)
        # self.drop0 = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 8, hidden_size*4)
        self.drop2 = nn.Dropout(dropout)

        #regular
        self.fc2 = nn.Linear(hidden_size*4, fc1_size)
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        #price
        self.price_fc2 = nn.Linear(hidden_size*4, fc1_size)
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

        if p1:
            x,x_d = x
            # x = self.batch_norm(x)
            x_d = x_d.float()
            x_d = x_d._replace(data=self.batch_norm_data(x_d.data))
            x = x.float()
            # x_d = x_d.float()
            x = x._replace(data=self.batch_norm(x.data))
            x_og = x

            x,hidden = self.gru(x, h)
            x = unpack_sequence(x)
            x_d = unpack_sequence(x_d)
            # x_og = unpack_sequence(x_og)
            outs = []
            for i,dog in enumerate(x):
                dog_simple = self.extra_1(x_d[i])
                dog = torch.cat((dog,dog_simple),dim=1)
                outs.append(dog)
            # print(f"{outs[0].shape=} ")
            return outs,hidden
        else:
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.relu(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.relu(x)
            x_e = self.drop2(x)
            #regular
            x = self.fc2(x_e)
            x_rl3 = self.relu(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)
            #price
            x_p = self.price_fc2(x_e)
            x_p_rl3 = self.relu(x_p)
            x_p = self.price_drop3(x_p_rl3)
            x_p = self.price_fc3(x_p)

            output = self.output_fn(x), x_rl3, self.output_fn(x_p),
            return output


class GRUNetv3_profit(nn.Module):
    def __init__(self,raceDB:Races):
        super(GRUNetv3_profit, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.3)

        self.fc0 = nn.Linear(8*3, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, output, output_p, prices):
        output = output.float().detach()
        output_p = output_p.float().detach()
        prices = prices.float()
        x = torch.cat([output,output_p,prices],dim=-1)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
class GRUNetv3_profit_testing2(nn.Module):
    def __init__(self,raceDB:Races):
        super(GRUNetv3_profit_testing2, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropoutfc1 = nn.Dropout(0.3)
        self.dropoutfc2 = nn.Dropout(0.3)
        self.dropoutfc3 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(2630)
        self.fc0 = nn.Linear(2630+8+8, 1280)
        self.fc1 = nn.Linear(1280, 1280)
        self.fc12  = nn.Linear(1280, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, output, output_p, prices):
        # output = self.softmax(output.float())
        # output_p = self.softmax(output_p.float())
        output = output.detach()
        # output = self.batch_norm(output.float())
        output = self.dropout2(output)
        output_p = self.softmax(output_p.detach())
        output_p = output_p.float().detach()
        # price_over_30_mask = prices > 30
        # price_over_30_mask = price_over_30_mask.to(float)*-1000
        # prices = (prices.float()**-1).nan_to_num(0,0,0)
        prices = prices**-1
        prices = prices.nan_to_num(0,0,0)
        x = torch.cat([output,output_p,prices],dim=-1).float()
        # x = self.batch_norm(x)
        if x.isnan().any():
            raise ValueError("NaN values detected in the input")
        x = self.relu(x)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.dropoutfc1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropoutfc2(x)
        x = self.fc12(x)
        x = self.relu(x)
        x = self.dropoutfc3(x)
        x = self.fc2(x)
        if x.isnan().any():
            raise ValueError("NaN values detected in the input and end of fc2")
        # x = x+price_over_30_mask
        x = self.softmax(x)
        return x

class GRUNetv3_profit_testing(nn.Module):
    def __init__(self,raceDB:Races):
        super(GRUNetv3_profit_testing, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.3)
        # self.batch_norm = nn.BatchNorm1d(8*3)
        self.fc0 = nn.Linear(8*3, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, output, output_p, prices):
        # output = self.softmax(output.float())
        # output_p = self.softmax(output_p.float())
        output = output.detach()
        output_p = self.softmax(output_p.detach())
        # output_p = output_p.float().detach()

        prices = prices.float().nan_to_num(0,0,0)
        price_over_30_mask = prices > 30
        price_over_30_mask = price_over_30_mask.to(float)*-1000
        x = torch.cat([output,output_p,prices],dim=-1)
        # x = self.batch_norm(x)
        if x.isnan().any():
            raise ValueError("NaN values detected in the input")
        x = self.relu(x)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if x.isnan().any():
            raise ValueError("NaN values detected in the input and end of fc2")
        x = x+price_over_30_mask
        x = self.softmax(x)
        return x

class GRUNetv3_profit_stacking(nn.Module):
    def __init__(self,raceDB:Races,num_models=5):
        super(GRUNetv3_profit_stacking, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.3)
        self.num_models = num_models
        for i in range(num_models):
            setattr(self, f'model_{i}', GRUNetv3_profit(raceDB))
            setattr(self, f'optim_{i}', optim.Adam(getattr(self, f'model_{i}').parameters(), lr=0.001, maximize=True))
        self.model_list = [getattr(self, f'model_{i}') for i in range(num_models)]
        self.optim_list = [getattr(self, f'optim_{i}') for i in range(num_models)]
        self.fc0 = nn.Linear(num_models*8, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, output, output_p, prices):
        # output = self.softmax(output.float())
        # output_p = self.softmax(output_p.float())
        outputs = []
        for i in range(self.num_models):
            output = getattr(self, f'model_{i}')(output,output_p,prices)
            outputs.append(output)
        x = torch.cat(outputs,-1).detach()
        x = self.relu(x)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class GRUNetv4_extra(nn.Module):
    def __init__(self,raceDB, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1,data_mask_size=None,model_number=0):
        super(GRUNetv4_extra, self).__init__()
        self.raceDB = raceDB
        self.h0 = nn.Parameter(torch.zeros(num_layers, hidden_size))
        self.hidden_dict = defaultdict(lambda: self.h0)


        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)
        

        self.batch_norm = nn.BatchNorm1d(input_size)
        self.batch_norm_data = nn.BatchNorm1d(data_mask_size)

        #extra data
        self.extra_1 = GRUNetv3_simple_extra_data(20,dropout,fc0_size,fc1_size)

        #p1
        self.fc0_p1 = nn.Linear(hidden_size+4*fc1_size,hidden_size)
        self.fc0_p1_drop = nn.Dropout(dropout)
        self.fc0_p2 = nn.Linear(hidden_size,hidden_size)
        self.fc0_p2_drop = nn.Dropout(dropout)
        self.fc0_p3 = nn.Linear(hidden_size,hidden_size)
        self.fc0_p3_drop = nn.Dropout(dropout)

        #p2
        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(((hidden_size+1*fc1_size) * 8)+70, hidden_size * 8)
        # self.fc0 = nn.Linear(((hidden_size) * 8)+70, hidden_size * 8)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 8, hidden_size*4)
        self.drop2 = nn.Dropout(dropout)

        #regular
        self.fc2 = nn.Linear(hidden_size*4, fc1_size)
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size
        self.model_number = model_number
        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise   

        self.output_cache = {}

    def reset_hidden(self):
        for dog in self.hidden_dict.keys():
            self.hidden_dict[dog] = self.h0.detach()


    # x represents our data
    def forward(self, x, x_d, dog_input,dogs, batch_races):
        if batch_races[0].raceid in self.output_cache.keys():
            # print('cached')
            return self.output_cache[batch_races[0].raceid]
        else:
            x_d = x_d.float()
            x = x.float()
            x_d = x_d._replace(data=self.batch_norm_data(x_d.data))
            x = x._replace(data=self.batch_norm(x.data))
            for dog in dogs:
                self.hidden_dict[dog].shape
            # print(type(dogs))
            hidden_in = torch.stack([self.hidden_dict[x] for x in dogs]).transpose(0,1)
            # hidden_in = self.h0.unsqueeze(1).repeat(1,x.batch_sizes[0],1)
            x,hidden = self.gru(x,hidden_in)
            hidden = hidden.transpose(0,1)
            x = unpack_sequence(x)
            x_d = unpack_sequence(x_d)
            outs = []
            for i,dog in enumerate(x):
                dog_simple = self.extra_1(x_d[i])
                dog = torch.cat((dog,dog_simple),dim=1)
                outs.append(dog)

            for i,dog in enumerate(dogs):
                self.hidden_dict[dog] = hidden[i].detach()

            for j,dog in enumerate(dog_input):
                [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,outs[j])]

            [setattr(race, 'hidden_in', torch.cat([race.race_dist]+[race.race_track]+[d.hidden_out for d in race.dogs])) for race in batch_races]

            x = torch.stack([r.hidden_in for r in batch_races])
            # print(f"p2 running model {self.model_number}")
            x = x.float()
            # x  = self.layer_norm2(x)
            x = self.relu0(x)
            # x = self.drop0(x)
            x = self.fc0(x)
            x = self.relu(x)
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.relu(x)
            x_e = self.drop2(x)
            #regular
            x = self.fc2(x_e)
            x_rl3 = self.relu(x)
            x = self.drop3(x_rl3)
            x = self.fc3(x)
            #price
            x_p = self.fc2(x_e)
            x_p_rl3 = self.relu(x_p)
            x_p = self.drop3(x_p_rl3)
            x_p = self.fc3(x_p)
            self.output = self.output_fn(x)

            output = self.output_fn(x).detach(), x_rl3.detach(), self.output_fn(x_p).detach(),

            self.output_cache[batch_races[0].raceid] = output

            return output

class GRUNetv4_stacking(nn.Module):
    def __init__(self,raceDB:Races, input_size, hidden_size,num_models,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1,data_mask_size=None):
        super(GRUNetv4_stacking, self).__init__()
        self.hidden_size = hidden_size

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_0 = nn.Linear(8*num_models,8*num_models)
        self.fc_1 = nn.Linear(8*num_models,8)

        self.fc_p0 = nn.Linear(8*num_models,8*num_models)
        self.fc_p1 = nn.Linear(8*num_models,8)

        for i in range(num_models):
            setattr(self, f'model_{i}', GRUNetv4_extra(raceDB,input_size,hidden_size,hidden,output,dropout,fc0_size,fc1_size,num_layers,data_mask_size,model_number=i))
        self.model_list:list[GRUNetv4_extra] = [getattr(self, f'model_{i}') for i in range(num_models)]
        self.optim_list = [torch.optim.Adam(model.parameters(), lr=0.0005) for model in self.model_list]
        self.scheduler_list = [torch.optim.lr_scheduler.ExponentialLR(optimizer,0.95) for optimizer in self.optim_list]
        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    def train_ensemble(self):
        self.train()
        [model.eval() for model in self.model_list]

    # x represents our data
    def forward(self, x, x_d, dog_input, dogs, batch_races, stacking=True):
        outputs = []
        relus = []
        output_ps = []
        if stacking:
            for i,model in enumerate(self.model_list):
                # print(f"running model {i}")
                output,relu,output_p = model.forward(x, x_d, dog_input,dogs, batch_races)
                outputs.append(output)
                relus.append(relu)
                output_ps.append(output_p)
            outputs = torch.cat(outputs,dim=-1).detach()
            # print(f"{outputs.shape=}")
            outputs = self.fc_0(outputs)
            outputs_relu = self.relu(outputs)
            outputs = self.dropout(outputs_relu)
            outputs = self.fc_1(outputs)

            output_ps = torch.cat(output_ps,dim=-1).detach()
            output_ps = self.fc_p0(output_ps)
            output_ps_relu = self.relu(output_ps)
            output_ps = self.dropout(output_ps_relu)
            output_ps = self.fc_p1(output_ps)

            return outputs, outputs_relu, output_ps         
        else:
            for i,model in enumerate(self.model_list):
                # print(f"running model {i}")
                output,relu,output_p = model.forward(x[i], x_d[i], dog_input[i],dogs[i], batch_races[i])
                outputs.append(output)
                relus.append(relu)
                output_ps.append(output_p)
            return outputs,relus,output_ps






