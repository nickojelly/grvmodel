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
# from rnn_classes import Dog, DogInput, Race, Races, GRUNet, smallGRUNet, smalll_lin_GRUNet, smalll_prelin_GRUNet
import rnn_classes
from rnn_classes import Races
from raceDB import build_dataset, build_pred_dataset
import importlib
import datetime
from model_saver import model_saver_linux as model_saver
import training_testing_gru
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence,pad_sequence, unpack_sequence, unpad_sequence
import logging

WANDB_MODE="offline"

def feature_selector(raceDB:Races, mask:torch.Tensor):
    pass
     
     

def model_pipeline(my_dataset=None,config=None,prev_model=None, sweep=True, model_state_dict=None, prev_model_file=None, prev_model_version='450'):
        stat_list = ['dist_last__1', 'box_last__1', 'speed_avg_1', 'split_speed_v1_1', 'split_speed_avg_1', 'split_margin_avg_1', 'margin_avg_1', 'margin_time_avg_1', 'RunHomeTime_1', 'run_home_speed_1', 'run_home_speed_v1_1', 'first_out_avg_1', 'pos_out_avg_1', 'post_change_avg_1', 'races_1', 'wins_1', 'wins_last_1', 'weight_', 'min_time_', 'min_split_time_', 'min_split_time_v1', 'last_start_price', 'last_start_prob']

        
        if my_dataset:
            dataset = my_dataset    
        else:
            dataset:Races = raceDB 




        # tell wandb to get started
        # torch._dynamo.config.verbose=True

        # raceDB.batch
        with wandb.init(project="runpod-gpu", config=config):
            #  access all HPs through wandb.config, so logging matches execution!
            wandb.define_metric("loss_val", summary="min")
            wandb.define_metric("accuracy", summary="max")
            wandb.define_metric("ROI < 30", summary="max")
            wandb.define_metric("relu roi", summary="max")
            wandb.define_metric("val_ROI < 30", summary="max")
            wandb.define_metric("val_loss_val", summary="min")
            
            config = wandb.config
            print(f"config = {config}")
            pprint.pprint(config)
            
            pprint.pprint(config['epochs'])
            print(config)

            stat_mask = [1,1,1]
            for stat in stat_list:

                stat_flag = config[stat]

                stat_mask = stat_mask+[stat_flag]

            stat_mask = torch.tensor(stat_mask).type(torch.bool).to(device)

            if 'batch_days' in config.keys():
                raceDB.create_test_split_date(config['training_date_end'])
                raceDB.create_dogs_test_split_date()
                raceDB.attach_races_to_dog_inputv2() 
                raceDB.reset_hidden()
                raceDB.create_batches(batch_days=config['batch_days'], end_date = config['training_date_end'], stat_mask=stat_mask)

            if config['input_type'] == 'basic':
                print('here')
                raceDB.batches['packed_x'] = raceDB.batches['packed_x_basic']
                raceDB.batches['packed_y'] = raceDB.batches['packed_y_basic']
                raceDB.batches['packed_v'] = raceDB.batches['packed_v_basic']
                input_size = raceDB.batches['packed_x'][0].data[0].shape[0]

            print(f"{input_size=}")

            


            raceDB.reset_hidden(num_layers=2, hidden_size=128)
        #   model = rnn_classes.GRUNetv3_LN(input_size,config['hidden_size'], num_layers=config['num_layers'],fc0_size=config['f0_layer_size'], fc1_size=config['f1_layer_size'])
            # model = rnn_classes.GRUNetv3_BN(input_size,config['hidden_size'], num_layers=config['num_layers'],fc0_size=config['f0_layer_size'], fc1_size=config['f1_layer_size'])
            model = rnn_classes.GRUNetv3_BN(input_size,128, num_layers=2,fc0_size=128, fc1_size=64)
            if model_state_dict:
                model.load_state_dict(model_state_dict)
            if prev_model_file!=None:
                model_name = prev_model_file
                model_loc = f"{model_name}_{prev_model_version}.pt"
                model_data = torch.load(model_loc,map_location=torch.device('cuda:0'))
                model.load_state_dict(model_data['model_state_dict'], strict=False)
                config['parent model'] = prev_model_file
                model = model.to(device)
                optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
                optimizer.load_state_dict(model_data['optim'])
                # optimizer.to(device)
            else:
                optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

            raceDB.to_cuda()
            optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
            criterion = nn.CrossEntropyLoss(reduction='none')

            model = model.to(device)

            # optimizer = optimizer.to(device)
            print(model)

            # and use them to train the model
            try:
                training_testing_gru.train_regular_v3(model, dataset, criterion, optimizer, 'na', config)
            except (KeyboardInterrupt) as error:
                print(error)
                print("finished Early")
                
            # dataset.create_hidden_states_dict()
            raceDB.create_hidden_states_dict_v2()
            model_saver(model, optimizer, 450, 0.1, raceDB.hidden_states_dict_gru_v6,raceDB.train_hidden_dict , model_name="long nsw new  22000 RUN")
            if sweep:
                # raceDB.reset_all_lstm_states
                raceDB.reset_hidden()
        
        return (model,dataset, optimizer)



if __name__=="__main__":

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        # os.chdir(r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\DATA")

        if torch.cuda.is_available():
                device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
                print("Running on the GPU")
        else:
                device = torch.device("cpu")
                print("Running on the CPU")


        WANDB_MODE="offline"
        os.getcwd()
        date = datetime.datetime.strptime("2022-08-01", "%Y-%m-%d").date()
        hidden_size = 32
        # states = ["SA", "VIC", "QLD", "NSW"]
        # states = ["NSW"]
        # states = ["VIC"]
        states=["NZ"]
        data_file = 'gru_inputs_new_simple_kitchen_sink_test.fth'
        data_file = 'gru_inputs_simple_kitchen_sink.fth'
        logging.basicConfig(filename='model_train_compile.log', encoding='utf-8', level=logging.DEBUG)
        logging.info(f"{states=}, {data_file=}")
        raceDB = build_dataset(data_file, hidden_size ,state_filter=states, margin_type='boosted_sftmin',v6=True)
        raceDB.create_new_weights_v2()

        # date = datetime.datetime.strptime("2022-08-01", "%Y-%m-%d").date()
        # date = "2022-08-01"
        # raceDB.create_test_split_date(date)

        # raceDB.create_dogs_test_split_date()

        # raceDB.attach_races_to_dog_inputv2() 

        # raceDB.create_batches(batch_days=180)

        wandb_config_static = {'hidden_size':128,
                                'stats':raceDB.stats_cols,
                                'races':states,
                                'datafile':data_file,
                                'latest_date':raceDB.latest_date,
                                'input_type':'basic',
                                'num_layers':2,
                                'batch_size': 750,
                                'dropout': 0.3,
                                'epochs': 20,
                                'learning_rate': 0.0001,
                                'optimizer': 'adamW',
                                'f0_layer_size':128,
                                'f1_layer_size':64}
        print(wandb_config_static)
        logging.info(f'load complete\n{wandb_config_static=}')

        

        raceDB.reset_hidden(num_layers=wandb_config_static['num_layers'], hidden_size=wandb_config_static['hidden_size'])

        # (model,dataset, optimizer) = model_pipeline(raceDB,config=wandb_config_static,sweep=False)

        sweep_config = {"method": "bayes"}

        metric = {"name": "val_loss_val.min", "goal": "minimize"}

        sweep_config["metric"] = metric


        parameters_dict = {
            "optimizer": {"value": "adamW"},
            "batch_days": {"values": [365]},
            # "batch_days": {"values": [90,180,365,550,10000]},
            # "f0_layer_size": {"values": [128]},
            # "f1_layer_size": {"values": [64]},
            # "dropout": {"values": [0.3]},
            "input_type": {"values": ['basic']},
            # "num_layers": {"values": [2]},
            # 'hidden_size':{'values':[128]},
            "len_data": {"value": len(raceDB.raceIDs)},
            # "stats":{"value": raceDB.stats_cols},
            "races":{"value": states},
            # "batch_size":{"value": 10},
            'dist_last__1': {'values': [0, 1]},
            'box_last__1': {'values': [0, 1]},
            'speed_avg_1': {'values': [0, 1]},
            'split_speed_v1_1': {'values': [0, 1]},
            'split_speed_avg_1': {'values': [0, 1]},
            'split_margin_avg_1': {'values': [0, 1]},
            'margin_avg_1': {'values': [0, 1]},
            'margin_time_avg_1': {'values': [0, 1]},
            'RunHomeTime_1': {'values': [0, 1]},
            'run_home_speed_1': {'values': [0]},
            'run_home_speed_v1_1': {'values': [0]},
            'first_out_avg_1': {'values': [0]},
            'pos_out_avg_1': {'values': [0, 1]},
            'post_change_avg_1': {'values': [0, 1]},
            'races_1': {'values': [0, 1]},
            'wins_1': {'values': [0, 1]},
            'wins_last_1': {'values': [0, 1]},
            'weight_': {'values': [0, 1]},
            'min_time_': {'values': [0, 1]},
            'min_split_time_': {'values': [0, 1]}, #works
            'min_split_time_v1': {'values': [0, 1]},
            'last_start_price': {'values': [0, 1]},
            'last_start_prob': {'values': [0, 1]}
        }

        sweep_config["parameters"] = parameters_dict

        parameters_dict.update(
            {
                "epochs": {"values": [500]},
                # "validation_split": {"value": 0.1},
                "training_date_end": {"values": 
                                      [
                                       #'2023-04-30',
                                       '2023-02-28',
                                       '2022-11-30',
                                       '2021-11-30'
                                      ]},
                "learning_rate": {"values": [0.0003, 0.0001]},
                # "label_smoothing": {"values": [0.01,0.0]},
                # "loss": {"values": [ "CEL"],},
            }
        )

        # parameters_dict.update(
        #     {

        # }
        # )

        


        sweep_id = wandb.sweep(sweep_config, project="Variable Sweeps")
        CUDA_LAUNCH_BLOCKING=1
        # wandb.agent(sweep_id , function=model_pipeline, count=100,project="Variable Sweeps")
        wandb.agent('fecofxxy', function=model_pipeline, count=100,project="Variable Sweeps")
        # wandb.agent('runpod GRU-sweeps/vq4tu6z9', function=model_pipeline, count=100)
