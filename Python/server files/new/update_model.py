import pandas as pd
import os
import torch
import torch.nn as nn
import torch

from random import randint
from rnn_classes import Dog, DogInput, Race, Races
from raceDB import build_dataset, build_pred_dataset
import datetime
from rnn_classes import GRUNetv3
# import val_model_external
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
import wandb
import logging
import torch.optim as optim
import pprint
import rnn_classes
import training_testing_gru
import training_testing_gru_double
from model_saver import model_saver_linux
import sys

DEVICE = 'cpu'

CONFIG = {'v6': {'states':'NSW',
				 'name':'nsw_model_relu',
				 'model_type':rnn_classes.GRUNetv3_LN,
				 'prev_model_file':'kind-wood-610',
				 'prev_model_version':'1300',
				 'test_func':training_testing_gru.test_model_v3,
				 'data_set':'gru_inputs_new_simple_v6_test.fth'
				 },
			'og_nz':{'states':'NZ',
				 'name':'nz_model',
				 'model_type':rnn_classes.GRUNetv3,
				 'prev_model_file':'bright-bee-555',
				 'prev_model_version':'1600',
				 'test_func':training_testing_gru.test_model_v3,
				 'data_set':'gru_inputs_new_simple_test.fth'
				 },
			'new_nz':{'states':'NZ',
				 'name':'nz_model',
				 'model_type':rnn_classes.GRUNetv3_BN_double,
				 'prev_model_file':'splendid-night-22',
				 'prev_model_version':'1100',
				 'test_func':training_testing_gru_double.test_model_v3,
				 'data_set':'gru_inputs_new_simple_test.fth'
				 },

			}

def load_db(states,hidden_size=128):

    date = datetime.datetime.strptime("2022-08-01", "%Y-%m-%d").date()
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    pred_db = f'prediction_inputs/testing new outs simple {today}.fth'
    raceDB = build_dataset(pred_db, hidden_size, state_filter=states, margin_type='boosted_sftmin',v6=True, date_filter=date)
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    raceDB.create_test_split_date(date)
    raceDB.create_dogs_test_split_date()
    raceDB.attach_races_to_dog_inputv2()

    return raceDB

def setup_model_v3(model_name,model_version, input_size, hidden_size, f0_size,f1_size):
    model = GRUNetv3(input_size,hidden_size,num_layers=2, fc0_size=f0_size, fc1_size=f1_size)
    prev_model = f"models/{model_name}_{model_version}.pt"
    prev_model_loaded = torch.load(prev_model,map_location=torch.device('cpu'))
    model.load_state_dict(prev_model_loaded['model_state_dict'])
    hidden_state_dict = prev_model_loaded['db_train']

    return model,hidden_state_dict

def model_pipeline(my_dataset,config=None,prev_model=None, sweep=True, model_state_dict=None, prev_model_file=None, prev_model_version='450', v6=False,model_config=None):
	if my_dataset:
		dataset = my_dataset    
	else:
		dataset = raceDB
    # tell wandb to get started
	with wandb.init(project="NEW GRU - updates -v7", config=config):	
		#  access all HPs through wandb.config, so logging matches execution!	
		wandb.define_metric("loss", summary="min")
		wandb.define_metric("test_accuracy", summary="max")
		wandb.define_metric("bfprofit", summary="max")
		wandb.define_metric("multibet profit", summary="max")
		
		config = wandb.config
		pprint.pprint(config)
		pprint.pprint(config.epochs)
		print(config)

		stat_list = ['dist_last__1', 'box_last__1', 'speed_avg_1', 'split_speed_v1_1', 'split_speed_avg_1', 'split_margin_avg_1', 'margin_avg_1', 'margin_time_avg_1', 'RunHomeTime_1', 'run_home_speed_1', 'run_home_speed_v1_1', 'first_out_avg_1', 'pos_out_avg_1', 'post_change_avg_1', 'races_1', 'wins_1', 'wins_last_1', 'weight_', 'min_time_', 'min_split_time_', 'min_split_time_v1', 'last_start_price', 'last_start_prob']
		
		# stat_mask = [1,1,1,1,1,1]
		# for stat in stat_list:

		# 	stat_flag = config[stat]

		# 	stat_mask = stat_mask+[stat_flag]

		# stat_mask = torch.tensor(stat_mask).type(torch.bool).to('cpu')

		len_test = len(raceDB.test_dog_ids)
		test_idx = range(0,len_test)
		raceDB.batches = {}
		raceDB.batches['packed_y'] = pack_sequence([torch.stack(n,0) for n in [[z.stats.to(DEVICE) for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]], enforce_sorted=False)
		# raceDB.batches['packed_y'] = pack_sequence([torch.stack(n,0) for n in [[z.stats.masked_select(stat_mask) for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]], enforce_sorted=False)
		input_size = raceDB.batches['packed_y'][0].data[0].shape[0]

		print(f"{input_size=}")

		model_class = model_config['model_type']
		print(model_class)
		test_func = model_config['test_func']

		model = model_class(input_size,config['hidden_size'], num_layers=config['num_layers'],fc0_size=config['f0_layer_size'], fc1_size=config['f1_layer_size'])

		model_name = prev_model_file
		model_loc = f"models/{model_name}_{prev_model_version}.pt"
		model_data = torch.load(model_loc,map_location=torch.device('cpu'))
		model.load_state_dict(model_data['model_state_dict'])
		raceDB.fill_hidden_states_dict_v2(model_data['db_train'])
		config['parent model'] = prev_model_file


		criterion = nn.CrossEntropyLoss(reduction='none')
		optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
		model = model.to(DEVICE)

		# raceDB.to_cpu()

		print(model)

		model.eval()
		test_func(model,raceDB, criterion,epoch=0, device='cpu')

		raceDB.create_hidden_states_dict_v2()
		model_name = model_config['name']
		
		model_saver_linux(model, optimizer, 450, 0.1, raceDB.hidden_states_dict_gru_v6,raceDB.train_hidden_dict , model_name=model_name)


	return (model,dataset, optimizer)	

if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	print(os.listdir())

	args = sys.argv

	model_config = CONFIG[args[1]]


	


	date = datetime.datetime.strptime("2022-11-30", "%Y-%m-%d").date()

	raceDB = build_dataset(f"/root/grv_model/db_update/DATA/{model_config['data_set']}", 128, state_filter=model_config['states'], margin_type='boosted_sftmin',v6=True, date_filter=date, device='cpu')

	raceDB.create_test_split_date(date)
	raceDB.create_dogs_test_split_date()
	raceDB.attach_races_to_dog_inputv2()
	# raceDB.create_batches()

	races = raceDB.get_test_input([1,2,3])
	print(races[0].race_dist)

	print(DEVICE)

	wandb_config_static = {'hidden_size':128,
                        'stats':raceDB.stats_cols,
                        # 'races':states,
                        # 'datafile':data_file,
                        # 'latest_date':raceDB.latest_date,
                        'input_type':'basic',
                        'num_layers':2,
                        'batch_size': 750,
                        'dropout': 0.3,
                        'epochs': 5000,
                        'learning_rate': 0.0001,
                        'optimizer': 'adamW',
                        'f0_layer_size':128,
                        'f1_layer_size':64,
                        'training_date_end':'2022-11-30',
                        'batch_days':365,
                        'dist_last__1': 1,#
                        'box_last__1':1,#
                        'speed_avg_1':1,#
                        'split_speed_v1_1':1,#
                        'split_speed_avg_1':0,#
                        'split_margin_avg_1':1,
                        'margin_avg_1':1,#
                        'margin_time_avg_1':0,#
                        'RunHomeTime_1':0,
                        'run_home_speed_1':0,
                        'run_home_speed_v1_1':0,
                        'first_out_avg_1':1,#
                        'pos_out_avg_1':0,#
                        'post_change_avg_1':1,#
                        'races_1':1,
                        'wins_1':1,
                        'wins_last_1':0,
                        'weight_':1,#
                        'min_time_':1,#
                        'min_split_time_':0, 
                        'min_split_time_v1':1,#
                        'last_start_price':1, #
                        'last_start_prob':0,
            }

	# raceDB.reset_hidden(num_layers=wandb_config_static['num_layers'], hidden_size=wandb_config_static['hidden_size'])

	# if v6:
	(model,dataset, optimizer) = model_pipeline(raceDB,config=wandb_config_static,sweep=False, prev_model_file=model_config['prev_model_file'], prev_model_version=model_config['prev_model_version'],v6=True,model_config=model_config)
	# else:
	# 	(model,dataset, optimizer) = model_pipeline(raceDB,config=wandb_config_static,sweep=False, prev_model_file='bright-bee-555', prev_model_version='1600')

