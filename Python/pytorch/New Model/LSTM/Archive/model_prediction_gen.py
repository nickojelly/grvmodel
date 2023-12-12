import pickle
import pandas as pd
import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm, trange
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
from random import randint
from rnn_classes import Dog, DogInput, Race, Races, GRUNet
from raceDB import build_dataset, build_pred_dataset
import importlib

def build_model_from_save(hidden_size, model_save_file):

    model_dict = torch.load(model_save_file)
    hidden_states_dict = model_dict['db']
    model_size = model_dict['model_state_dict']['gru1.weight_ih'].shape[1]
    model = GRUNet(model_size, hidden_size, output = 'softmax')
    model.load_state_dict(model_dict['model_state_dict'])

    return (model, hidden_states_dict)

def flatten(l):
    return [item for sublist in l for item in sublist]

def generate_predictions(model, predDB):

    batch_size = 10
    len_test = len(predDB.raceIDs)-batch_size
    last = 0



    output_dict = {}
    output_dict['race_id'] = []
    output_dict['track'] = []
    output_dict['race_date'] = []
    output_dict['race_time'] = []
    output_dict['dog_name'] = []
    # output_dict['dog_box'] = []
    output_dict['probabilty'] = []

    with torch.no_grad():
        for i in trange(0,len_test,batch_size, leave=False):
            races_idx = range(last,last+batch_size)
            last = i
            race = predDB.get_race_input(races_idx)
            print(race)
            print([r.raceid for r in race])
            X = race
            # y = torch.stack([x.classes for x in race])
            output = model(X)

            output_dict['race_id'].extend(flatten([[x.raceid]*8 for x in race]))
            output_dict['track'].extend(flatten([[x.track_name]*8 for x in race]))
            output_dict['race_date'].extend(flatten([[x.race_date]*8 for x in race]))
            output_dict['race_time'].extend(flatten([[x.race_time]*8 for x in race]))
            output_dict['dog_name'].extend([x.dog.dog_name for x in race for x in x.dogs])
            # output_dict['dog_box'].extend([x. for x in race])
            output_dict['probabilty'].extend([x.item() for item in output for x in item])

            break

    df_pred = pd.DataFrame(output_dict)
    
    return df_pred


if __name__ == "__main__":
    model_file = r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\pytorch\New Model\savedmodel\test NZ GRU saver\test NZ GRU saver_450.pt"
    hidden_size = 64

    (model, hidden_states_dict) = build_model_from_save(64, model_file)
    print(model)

    # pred_db_file = open( r'C:\Users\Nick\Documents\GitHub\grvmodel\Python\DATA\prediction validation 2023-01.npy', 'rb')
    # predDB = build_dataset(pred_db_file, hidden_size)

    pred_db_file =  open(r'C:\Users\Nick\Documents\GitHub\grvmodel\Python\Model Prediction\pred_df 2023-01-12.npy', 'rb')
    predDB = build_pred_dataset(pred_db_file, hidden_size)
    predDB.fill_hidden_states_from_dict(hidden_states_dict)
    predDB.to_cpu()


    df = generate_predictions(model, predDB)
    df.to_csv('test_preds_out_after_db_load.csv')

