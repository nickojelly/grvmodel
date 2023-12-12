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
from model_saver import model_saver_linux
import sys

DEVICE = 'cpu'


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

def model_pipeline(my_dataset,config=None,prev_model=None, sweep=True, model_state_dict=None, prev_model_file=None, prev_model_version='450', v6=False):
	if my_dataset:
		dataset = my_dataset    
	else:
		dataset = raceDB
    # tell wandb to get started
	with wandb.init(project="NEW GRU - updates", config=config):	
		#  access all HPs through wandb.config, so logging matches execution!	
		wandb.define_metric("loss", summary="min")
		wandb.define_metric("test_accuracy", summary="max")
		wandb.define_metric("bfprofit", summary="max")
		wandb.define_metric("multibet profit", summary="max")
		
		config = wandb.config
		pprint.pprint(config)
		pprint.pprint(config.epochs)
		print(config)

		len_test = len(raceDB.test_dog_ids)
		test_idx = range(0,len_test)
		raceDB.batches = {}
		raceDB.batches['packed_y'] = pack_sequence([torch.stack(n,0) for n in [[z.stats.to(DEVICE) for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]], enforce_sorted=False)
		input_size = raceDB.batches['packed_y'][0].data[0].shape[0]

		print(f"{input_size=}")


		# raceDB.reset_hidden(num_layers=config['num_layers'], hidden_size=config['hidden_size'], device='cpu')
		if v6:
			model = rnn_classes.GRUNetv3_LN(input_size,config['hidden_size'], num_layers=config['num_layers'],fc0_size=config['f0_layer_size'], fc1_size=config['f1_layer_size'])
		else:
			model = rnn_classes.GRUNetv3(input_size,config['hidden_size'], num_layers=config['num_layers'],fc0_size=config['f0_layer_size'], fc1_size=config['f1_layer_size'])

		if model_state_dict:
			model.load_state_dict(model_state_dict)
		if prev_model_file!=None:
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
		training_testing_gru.validate_model_v3(model,raceDB, criterion,epoch=0, device='cpu')

		raceDB.create_hidden_states_dict_v2()
		if v6:
			model_name = 'nsw_model_relu'
		else:
			model_name = 'nz_model'
		model_saver_linux(model, optimizer, 450, 0.1, raceDB.hidden_states_dict_gru_v6,raceDB.train_hidden_dict , model_name=model_name)


	return (model,dataset, optimizer)	

if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	print(os.listdir())

	args = sys.argv

	if 'v6' in args:
		v6 = True
		states = ['VIC']
	else:
		v6 = False
		states = ['NZ']
	


	date = datetime.datetime.strptime("2022-08-01", "%Y-%m-%d").date()

	if v6:
		raceDB = build_dataset("/root/grv_model/db_update/DATA/gru_inputs_new_simple_v6_test.fth", 128, state_filter=states, margin_type='boosted_sftmin',v6=True, date_filter=date, device='cpu')
	else:
		raceDB = build_dataset("/root/grv_model/db_update/DATA/gru_inputs_new_simple_test.fth", 128, state_filter=states, margin_type='boosted_sftmin',v6=True, date_filter=date, device='cpu')

	raceDB.create_test_split_date(date)
	raceDB.create_dogs_test_split_date()
	raceDB.attach_races_to_dog_inputv2()
	# raceDB.create_batches()

	races = raceDB.get_test_input([1,2,3])
	print(races[0].race_dist)

	print(DEVICE)

	wandb_config_static = {'hidden_size':128,
						'stats':raceDB.stats_cols,
						'races':states,
						'input_type':'basic',
						'num_layers':2,
						'batch_size': 750,
						'dropout': 0.3,
						'epochs': 10_000,
						'learning_rate': 0.0008,
						'optimizer': 'adamW',
						'f0_layer_size':128,
						'f1_layer_size':64}

	# raceDB.reset_hidden(num_layers=wandb_config_static['num_layers'], hidden_size=wandb_config_static['hidden_size'])

	if v6:
		(model,dataset, optimizer) = model_pipeline(raceDB,config=wandb_config_static,sweep=False, prev_model_file='resilient-silence-668', prev_model_version='1500',v6=True)
	else:
		(model,dataset, optimizer) = model_pipeline(raceDB,config=wandb_config_static,sweep=False, prev_model_file='bright-bee-555', prev_model_version='1600')

