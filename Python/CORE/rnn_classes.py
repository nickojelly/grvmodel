import torch.nn as nn
import torch
from operator import itemgetter
import operator
from random import randint
from tqdm.notebook import tqdm, trange
import torch.nn.functional as F


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
        self.lstmCellh = None
        self.lstmCellc = None
        self.box = dog_box
        
    def lstm_i(self, hidden_state):
        self.gru_cell = hidden_state
        self.visited = self.visited + 1

    def lstm_2(self, hidden_state):
        self.gru_cell = hidden_state
        self.visited = self.visited + 1

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
        self.races[raceid] = DogInput(self.dogid, raceid, stats, self, box, torch.ones(self.hidden_size), bfsp, sp) #this is the change
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

    def lstm_input(self, pred=False):
        if pred:
            print('pred')
        else:
            l_input = [x.gru_cell for x in self.dogs]
        return l_input

    def lstm_detach(self):
        [x.detach_state for x in self.dogs]

    def list_dog_ids(self):
        dogs_l = [x.dog.dogid for x in self.dogs]
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
            dog.lstm_o(hs) #.clone())
            
    def add_weights(self, weights):
        self.weights = weights

    def add_win_weight(self):
        pass

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

    def create_test_split(self):
        self.train_race_ids = []
        self.test_race_ids = []
        for i,r in enumerate(self.raceIDs):
            if i%10>(8):
                self.test_race_ids.append(r)
            else:
                self.train_race_ids.append(r)

    def create_test_split_date(self, start_date):
        self.train_race_ids = []
        self.test_race_ids = []
        test,train = 0,0
        for i,r in enumerate(self.raceIDs):
            race_date = self.racesDict[r].race_date
            if race_date>start_date:
                self.test_race_ids.append(r)
                test += 1
            else:
                self.train_race_ids.append(r)
                train += 1
        print(f"Train examples {train}, Test examples {test}")

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

    def get_race_classes(self, idx):
        raceidx = self.raceIDs[idx]
        classes = [x for x in self.raceDict[raceidx].classes]
        return classes
        
    def get_race_weights(self, races):
        print("weights")

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

    def detach_given_hidden_states(self, races):
        for race in races:
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
        filled, empty = 0, 0
        for race in tqdm(self.racesDict.values()):
            race_id = race.raceid
            for dog in race.dogs:
                dog_id = dog.dogid
                if dog_id==-1:
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
        print(f"{filled =}\n{empty  =}\n{filled/(filled+empty)}")

    def to_cuda(self):
        for race in self.racesDict.values():
            race_id = race.raceid
            for dog in race.dogs:
                dog.gru_cell = dog.gru_cell.to('cuda:0')

    def to_cpu(self):
        for race in self.racesDict.values():
            race_id = race.raceid
            race.full_input = race.full_input.to('cpu')

            for dog in race.dogs:
                dog.gru_cell = dog.gru_cell.to('cpu')
                dog.stats = dog.stats.to('cpu')

    def race_prices_to_prob(self):
        for r in self.racesDict.values():
            r.implied_prob = [1/(x+0.0001) for x in r.prices]

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size,output='raw', dropout=0.3):
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

        if output =='raw':
            self.output_fn = nn.Identity()
        elif output =='softmax':
            self.output_fn = nn.Softmax(dim=1)
        elif output =='log_softmax':
            self.output_fn = nn.LogSoftmax(dim=1)
        else:
            raise

    # x represents our data
    def forward(self, race: Race)  :
        x = torch.stack([r.full_input.float() for r in race])#.detach()

        # creates list of LSTM data
        hidden_state_in = [list(i) for i in zip(*[r.lstm_input() for r in race])]

        # creates list of tensors for lstm Cells
        hCell = [torch.stack([x for x in y]) for y in hidden_state_in]

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
        xhh = torch.cat((h1, h2, h3, h4, h5, h6, h7, h8), dim=1)
        xr1 = self.rl1(xhh)
        xd1 = self.drop1(xr1)
        xh = self.fc2(xd1)
        xd2 = self.drop2(xh)
        xr2 = self.rl2(xd2)
        xf = self.fc3(xr2)

        # output = F.softmax(xf, dim=1)
        # output = self.lsftmax(xf)
        output = self.output_fn(xf)
        return output
        
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3, output='raw'):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(input_size, hidden_size)
        self.lstm3 = nn.LSTMCell(input_size, hidden_size)
        self.lstm4 = nn.LSTMCell(input_size, hidden_size)
        self.lstm5 = nn.LSTMCell(input_size, hidden_size)
        self.lstm6 = nn.LSTMCell(input_size, hidden_size)
        self.lstm7 = nn.LSTMCell(input_size, hidden_size)
        self.lstm8 = nn.LSTMCell(input_size, hidden_size)
        self.rl1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size * 8, 64)
        self.drop2 = nn.Dropout(dropout)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 8)
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
            #r.lstm_detach()
        xhh = torch.cat((h1,h2, h3, h4, h5, h6, h7, h8), dim=1)
        xr1 = self.rl1(xhh)
        xd1 = self.drop1(xr1)
        xh = self.fc2(xd1)
        xd2 = self.drop2(xh)
        xr2 = self.rl2(xd2)
        xf = self.fc3(xh)

        output = self.output_fn(xf)
        return output
