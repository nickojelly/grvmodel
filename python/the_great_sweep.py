import pickle
import pandas as pd
import os
import setup

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
# from rnn_classes import Dog, DogInput, Race, Races, GRUNet, smallGRUNet, smalll_lin_GRUNet, smalll_prelin_GRUNet
import rnn_tools.rnn_classes as rnn_classes
from rnn_tools.raceDB import build_dataset
import importlib
import datetime
from rnn_tools.model_saver import model_saver, model_saver_wandb
import rnn_tools.training_testing_gru as training_testing_gru
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence,pad_sequence, unpack_sequence, unpad_sequence
import rnn_tools.training_testing_gru_double as training_testing_gru_double
import rnn_tools.training_testing_lstm as training_testing_lstm
from goto_conversion import goto_conversion

import rnn_tools.training_testing_gru_extra_data as training_testing_gru_extra_data
import rnn_tools.training_testing_gru_ensemble as training_testing_gru_ensemble

stat_list_dict = {'_dist_last__1':1,
'_box_last__1':1,
'_speed_avg_1':1,
'_split_speed_avg_1':1,
'_split_margin_avg_1':1,
'_margin_avg_1':1,
'_margin_time_avg_1':1,
'_RunHomeTime_1':1,
'_run_home_speed_1':1,
'_first_out_avg_1':1,
'_pos_out_avg_1':1,
'_post_change_avg_1':1,
'_races_1':1,
'_wins_1':1,
'_wins_last_1':1,
'_weight_':1,
'_min_time_':1,
'_min_split_time_':1,
'_last_start_price':1,
'_last_start_prob':1,
}

def build_raceDB(start_date, state_filter) -> rnn_classes.Races:
    os.getcwd()
    import rnn_tools.raceDB
    importlib.reload(rnn_tools.raceDB)
    # os.chdir(r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\DATA")                                                                                                                                                                                                                                                                                                                             
    date = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d").date()
    hidden_size = 32
    data_file = './data/gru_inputs_kitchen_sink_new_not_simple.fth'
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()

    raceDB = rnn_tools.raceDB.build_dataset(data_file, hidden_size,date_filter=start_date,state_filter=state_filter, margin_type='boosted_sftmin',v6=True)
    raceDB.create_new_weights_v2()
    return raceDB

def model_pipeline(my_dataset,config=None,prev_model=None, sweep=True, model_state_dict=None, prev_model_file=None, prev_model_version='450'):
    raceDB = my_dataset
    device = 'cuda:0'
    if my_dataset:
      dataset = my_dataset    
    else:
      dataset = raceDB
    # tell wandb to get started
    with wandb.init(project="NEW GRU V7 Reporting", config=config,save_code=False):
      #  access all HPs through wandb.config, so logging matches execution!
      wandb.define_metric("loss_val", summary="min")
      wandb.define_metric("accuracy", summary="max")
      wandb.define_metric("ROI < 30", summary="max")
      wandb.define_metric("val_ROI < 30", summary="max")
      wandb.define_metric("relu roi", summary="max")
      
      config = wandb.config
      pprint.pprint(config)
      pprint.pprint(config.epochs)
      print(config)
      wandb.run.log_code('rnn_tools/')
      # input_size = raceDB.get_race_input([0,1])[0].full_input.shape[0] #create fix so messy
      print(config.input_type)

      for i in config.stat_list_dict.values():
        print(i)

      reg_stat_mask = []
      for stat,flag in config.stat_list_dict.items():
          stat_flag = flag
          reg_stat_mask = reg_stat_mask+[stat_flag]

      print(reg_stat_mask)
      stat_mask = [1]*6+reg_stat_mask+[0]*80
      data_mask = [0]*26+[1]*20+[1]*20+[1]*20+[1]*20 # Reg, Dist, Box, T_box, T_dist
      data_mask = [1]*6+reg_stat_mask+[0]*20+[0]*20+[0]*20+[0]*20 # Reg, Dist, Box, T_box, T_dist
      data_mask_size = sum(data_mask)
      print(f"{data_mask_size=}")
      stat_mask = torch.tensor(stat_mask).type(torch.bool).to(device)
      data_mask = torch.tensor(data_mask).type(torch.bool).to(device)
      # stat_mask = torch.ones_like(stat_mask).type(torch.bool).to(device)

      print(stat_mask)
      print(stat_mask.shape)

      if 'batch_days' in config.keys():
        pass
        raceDB.create_test_split_date(config['training_date_end'],val_date='2023-07-01')
        raceDB.create_dogs_test_split_date()
        raceDB.attach_races_to_dog_inputv2() 
        raceDB.reset_hidden()
        raceDB.create_batches(batch_days=config['batch_days'], end_date = config['training_date_end'], stat_mask=stat_mask,data_mask=data_mask)

      if config['input_type'] == 'basic':
          print('here')
          raceDB.batches['packed_x'] = raceDB.batches['packed_x_basic']
          raceDB.batches['packed_y'] = raceDB.batches['packed_y_basic']
          raceDB.batches['packed_v'] = raceDB.batches['packed_v_basic']
          input_size = raceDB.batches['packed_x'][0].data[0].shape[0]

      # for race in raceDB.racesDict.values():
      #     for dog_input in race.dogs:
      #       dog_input.stats_masked = dog_input.stats.masked_select(stat_mask) 


      print(f"{input_size=}")
      config['stat_mask_tensor'] = torch.tensor(stat_mask, dtype=torch.uint8).to(device)     
      raceDB.reset_hidden(num_layers=config['num_layers'], hidden_size=config['hidden_size'])
      model = rnn_classes.GRUNetv3_extra(input_size,config['hidden_size'], num_layers=config['num_layers'],fc0_size=config['f0_layer_size'], fc1_size=config['f1_layer_size'],data_mask_size=data_mask_size)
      optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
      
      if model_state_dict:
        model.load_state_dict(model_state_dict)
      if prev_model_file!=None:
        print(f"Loading model {prev_model_file}, version {prev_model_version}")
        model_name = prev_model_file
        model_loc = f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/{model_name}_{prev_model_version}.pt"
        model_data = torch.load(model_loc,map_location=torch.device('cuda:0'))
        print(model_data['model_state_dict'].keys())
        model.load_state_dict(model_data['model_state_dict'], strict=True)
        config['parent model'] = prev_model_file
        raceDB.fill_hidden_states_dict_v2(model_data['db_train'])
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        # optimizer.load_state_dict(model_data['optim'])
        # optimizer.to(device)
      else:
        optimizer = optim.RAdam(model.parameters(), lr=config['learning_rate'])

      raceDB.to_cuda()
      criterion = nn.CrossEntropyLoss(reduction='none')

      model = model.to(device)
      print(model)

      # and use them to train the model
      # wandb.watch(model, log='all')
      try:
        # train_double_loss_regular(model, dataset, criterion, optimizer, scheduler, config, crit2=custom_l2)
        training_testing_gru_extra_data.train_double_v3(model, dataset, criterion, optimizer, 'na', config)
        # hidden_state_init = model.h0
        # raceDB.reset_hidden_w_param(hidden_state_init,num_layers=2, hidden_size=config['hidden_size'])
        # raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(256+64)).to('cuda:0')
        
        # training_testing_gru_extra_data.test_model_v3(model, dataset, criterion=criterion,config=config)
        # raceDB.reset_hidden_w_param(hidden_state_init,num_layers=2, hidden_size=config['hidden_size'])
        # raceDB.dogsDict['nullDog'].input.hidden_out = (-torch.ones(256+64)).to('cuda:0')
        # training_testing_gru_extra_data.validate_model_v3(model, dataset, criterion=criterion,config=config)
      except (KeyboardInterrupt) as error:
        print(error)
        print("finished Early")
        
      # dataset.create_hidden_states_dict()
      raceDB.create_hidden_states_dict_v2()
      model_saver_wandb(model, optimizer, 450, 0.1, raceDB.hidden_states_dict_gru_v6,raceDB.train_hidden_dict , model_name="long nsw new  22000 RUN")
      if sweep:
        # raceDB.reset_all_lstm_states
        raceDB.reset_hidden()
    


    # and test its final performance
    #test(model, test_loader)

    return (model,dataset, optimizer)
def model_runner():
    run = wandb.init()

    # Access the configuration parameters
    config = run.config

    print(config)
    print(config['start_date'])
    # asda
    raceDB = build_raceDB(config['start_date'], config['state_filter'])
    
    _ = model_pipeline(raceDB,config=config)

if __name__ == "__main__":
   
    stat_list_dict = {'_dist_last__1':1,
    '_box_last__1':1,
    '_speed_avg_1':1,
    '_split_speed_avg_1':1,
    '_split_margin_avg_1':1,
    '_margin_avg_1':1,
    '_margin_time_avg_1':1,
    '_RunHomeTime_1':1,
    '_run_home_speed_1':1,
    '_first_out_avg_1':1,
    '_pos_out_avg_1':1,
    '_post_change_avg_1':1,
    '_races_1':1,
    '_wins_1':1,
    '_wins_last_1':1,
    '_weight_':1,
    '_min_time_':1,
    '_min_split_time_':1,
    '_last_start_price':1,
    '_last_start_prob':1,
    }

    parameters_dict = {
        'hidden_size': {'values': [256]},
        'input_type': {'values': ['basic']},
        'num_layers': {'values': [2]},
        'dropout': {'values': [0.3]},
        'epochs': {'values': [250]},
        'learning_rate': {'values': [0.001]},
        'optimizer': {'values': ['adamW']},
        'f0_layer_size': {'values': [128]},
        'f1_layer_size': {'values': [64]},
        'training_date_end': {'values': ['2023-01-01']},
        'notes': {'values': ['GRU, with basic add on data looped in']},
        'stat_list_dict': {'values': [stat_list_dict]},
        'start_date': {'values': ['2020-01-01','2021-01-01','2022-01-01']},
        'batch_days': {"values": [90,180,365]},
        'state_filter': {'values': ['NZ','NSW','VIC','QLD','SA','WA','TAS','NT']},
        }

    sweep_config = {
        "method": "grid",
        "metric": {
            "name": "ROI < 30",
            "goal": "maximize"
        },
    }

    # Add static parameters to the sweep config
    sweep_config["parameters"] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="Big Sweep")

    wandb.agent('9jcllg6g', function=model_runner, count=1000)


