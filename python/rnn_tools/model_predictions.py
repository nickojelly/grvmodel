import pandas as pd
import os
import torch
import torch.nn as nn
import torch

from random import randint
from rnn_classes import Dog, DogInput, Race, Races, GRUNet
from raceDB import build_dataset, build_pred_dataset
import datetime
from rnn_classes import GRUNetv3
# import val_model_external
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence

import logging

DEVICE = 'cpu'

def v6_forward_pass_test(raceDB:Races, model:GRUNetv3, basic=False):
    sft_max = nn.Softmax(dim=-1)
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        len_test = len(raceDB.test_dog_ids)
        test_idx = range(0,len_test)

        dogs = [x for x in  raceDB.test_dogs.values()]  #[Dog]
        print(f"length of dogs {len(dogs)}")
        dog_input = [inner for inner in [x for x in raceDB.get_dog_test(test_idx)]] #[[DogInput]]
    
        for dog in dogs:
            dog.hidden = dog.hidden.to(DEVICE)

        if basic:
            X = pack_sequence([torch.stack(n,0) for n in [[z.stats.to(DEVICE) for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]], enforce_sorted=False)
        else:
            X = pack_sequence([torch.stack(n,0) for n in [[z.full_input.to(DEVICE) for z in inner] for inner in [x for x in raceDB.get_dog_test(test_idx)]]], enforce_sorted=False)

        hidden_in = torch.stack([x.hidden for x in dogs]).transpose(0,1)

        print(hidden_in.shape)

        output,_ = model(X,h=hidden_in)

        for i,dog in enumerate(dog_input):
            dog_outputs = output[i]
            for j,di in enumerate(dog):
                di.hidden_out = dog_outputs[j]

        raceDB.margin_from_dog_to_race_v3(mode='test')

        len_test = len(raceDB.test_race_ids)
        test_idx = range(0,len_test)

        race = raceDB.get_test_input(test_idx)

        X = torch.stack([r.hidden_in for r in race])

        output,relu = model(X, p1=False)

        relu = torch.sum(abs(relu),dim=1)

        softmax_preds = sft_max(output).to('cpu')

        test_races = raceDB.get_test_input(test_idx)

        x = race

        
        relu_out = [r for s in  [[r]*8 for r in relu.tolist()] for r in s]
        outs_list = [item for sublist in softmax_preds.tolist() for item in sublist]
        dogs = [dog.dog.dog_name for sublist in [r.dogs for r in x] for dog in sublist]
        dog_hidden = [dog.dog.hidden_filled for sublist in [r.dogs for r in x] for dog in sublist]
        dog_last_run_date = [dog.dog.last_race_date for sublist in [r.dogs for r in x] for dog in sublist]
        box = [r for s in [[1,2,3,4,5,6,7,8] for r in x] for r in s]
        dogId = [dog.dog.dogid for sublist in [r.dogs for r in x] for dog in sublist]
        date = [r for s in [[r.race_date]*8 for r in x] for r in s]
        times = [r for s in [[r.race_time]*8 for r in x] for r in s]
        rid = [r for s in [[r.raceid]*8 for r in x] for r in s]
        track = [r for s in [[r.track_name]*8 for r in x] for r in s]
        rnum = [r for s in [[int(r.race_num)]*8 for r in x] for r in s]
        outs = pd.DataFrame(data = {"raceid":rid, 
                                    "track":track,
                                    'date':date,
                                    "racetime":times,
                                    "dogid":dogId,
                                    'dog_hidden':dog_hidden,
                                    'dog_last_run_date':dog_last_run_date,
                                    'relu_sum':relu_out,
                                    "conf":outs_list,
                                    "box":box,
                                    "dogs":dogs,
                                    "race_num":rnum })
        
        outs['pred_price'] = outs['conf'].apply(lambda x: 1/(x))
        outs = outs.sort_values(['track','race_num','box'])

        
    return outs

def load_db(states,hidden_size=128):

    date = datetime.datetime.strptime("2023-04-08", "%Y-%m-%d").date()
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    pred_db = f'prediction_inputs/testing new outs simple {today}.fth'
    raceDB = build_dataset(pred_db, hidden_size ,states, device='cpu')
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
    hidden_state_dict = prev_model_loaded['db']

    return model,hidden_state_dict

if __name__ == '__main__':

    print(os.listdir())

    states = ['NSW']

    raceDB = load_db(states, hidden_size=128)

    model_name = 'spring-dust-518'
    model,hidden_state_dict = setup_model_v3(model_name,1200, 20,128,128,64)

    logging.basicConfig(filename='model_prediction.log', encoding='utf-8', level=logging.DEBUG)

    raceDB.fill_hidden_states_dict_v2(hidden_state_dict)

    today_time = datetime.datetime.today().strftime('%Y-%m-%d %H_%M_%S')
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    today_date = datetime.datetime.today().date()

    logging.info(f'--- {today} --- \n')

    logging.info(f'Creating model predictions for {states}  with {model_name}')


    outs = v6_forward_pass_test(raceDB, model, basic=True)

    today = datetime.datetime.today().strftime('%Y-%m-%d')
    today_date = datetime.datetime.today().date()

    outs['Model Name'] = model_name
    outs = outs#[ outs['track'].isin(['Richmond', 'Wentworth Park'])]
    outs_1 = outs#[outs['date']==today_date]

    outs_1.to_pickle(f'./model_outputs/output {model_name} {today_time}.npy')
    outs_1.to_pickle(f'//root/grv_model/flumine_betting/inputs/output {today_time}.npy')
    outs_1.to_excel(f'./model_outputs/output {model_name} -{today_time}.xlsx')
