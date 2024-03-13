import pickle
import pandas as pd
import os
import torch
import betfairlightweight
from betfairlightweight import filters
from datetime import datetime
from datetime import timedelta
from dateutil import tz
import math
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
import fasttrack as ft
import importlib
import featurecreations
from tqdm.notebook import tqdm
from sklearn.preprocessing import OneHotEncoder
from ft_sec_key import SECKEY
import logging
import concurrent.futures
import numpy as np


def update_results_data_week(start_date, end_date):
    seckey = SECKEY
    greys = ft.Fasttrack(seckey)

    race_details, dog_results = greys.getRaceResults(start_date.strftime('%Y-%m-%d'), dt_end=end_date.strftime('%Y-%m-%d'))
    print(f"race detail cols = {race_details.columns}, dog detail cols = {dog_results.columns}")
    race_details = race_details.rename(columns={'@id':'RaceId'})
    dog_results = dog_results.rename(columns={'@id':'DogId'})
    all_results = dog_results.merge(race_details, how='left', on='RaceId')
    all_results = all_results[all_results['Place']!=None]
    return all_results

def update_results_data():
    today = datetime.today() + timedelta(days=1)
    lastweek = datetime.today() - timedelta(days=500)

    weeks = np.arange(lastweek, today, timedelta(days=7)).astype(datetime)
    weeks = np.append(weeks, today)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        all_results = list(executor.map(update_results_data_week, weeks[:-1], weeks[1:]))

    all_results = pd.concat(all_results)
    return all_results

def update_basic_form_data_week(start_date, end_date):
    seckey = SECKEY
    greys = ft.Fasttrack(seckey)

    race_details, dog_results = greys.getBasicFormat(start_date.strftime('%Y-%m-%d'), dt_end=end_date.strftime('%Y-%m-%d'))
    race_details['RaceId'] = race_details['@id']
    dog_results['DogId'] = dog_results['@id']
    dog_results['box'] = dog_results['RaceBox']
    basic_form = dog_results.merge(race_details,how='left', on='RaceId')[['@id_x', '@id_y', 'DogGrade']]
    basic_form = basic_form.rename(columns={'@id_x':'DogId','@id_y':'RaceId'})
    return basic_form

def update_basic_form_data():
    today = datetime.today() + timedelta(days=1)
    lastweek = datetime.today() - timedelta(days=500)

    weeks = np.arange(lastweek, today, timedelta(days=7)).astype(datetime)
    weeks = np.append(weeks, today)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        basic_forms = list(executor.map(update_basic_form_data_week, weeks[:-1], weeks[1:]))

    all_races = pd.read_pickle(r'./DATA/all basic form merged.npy')
    all_races = pd.concat([all_races] + basic_forms)
    all_races.drop_duplicates(subset=['DogId','RaceId'],keep='last')
    all_races.to_pickle(r'./DATA/all basic form merged.npy')

    return all_races


if __name__ == "__main__":

    betfair_sp_file = (
        r"./DATA/df_betfairSP.fth"
    )
    previous_results_file = (
        r"./DATA/all-results-db.npy"
    )
    split_dist_file = (
        r"./DATA/split_dist_ft_new.csv"
    )
    # prev_full_details = r"results-df-merged-prices_cut.npy"
    # prev_full_details = r"results-df-merged-prices_cut.csv"
    prev_full_details = r"./DATA/results-df-merged-prices_cut.fth"

    update = True

    if update:
        dog_results = update_results_data()
        basic_form = update_basic_form_data()
        dog_results = dog_results.merge(basic_form, how='left', on=['RaceId', 'DogId'])
        dog_results = dog_results.dropna(subset=['track_code'])
        dog_results.to_pickle(r'./DATA/ft_races.npy')
        dog_results_prev = pd.read_feather(r'./DATA/ft_races_ALL.fth')
        dog_results_all = pd.concat([dog_results_prev,dog_results]).drop_duplicates(subset=['DogId','RaceId'],keep='last').reset_index(drop=True).to_feather(r'./DATA/ft_races_ALL.fth')
    else:
        dog_results = pd.read_feather(r'./DATA/ft_races_ALL.fth')
    # dog_results = pd.read_pickle(r'./DATA/ft_races.npy')

    print(f"Latest date = {pd.to_datetime(dog_results.date).max()}")

    logging.basicConfig(filename='database_updater.log', encoding='utf-8', level=logging.DEBUG)
    today = datetime.today().strftime('%Y-%m-%d')
    logging.info(f'--- {today} --- \n')
    logging.info(f"Updating Database, Latest date = {pd.to_datetime(dog_results.date).max()}")


    print(dog_results.columns)
    print(dog_results.Track.value_counts())
    print(dog_results.track_code.value_counts())
    print(dog_results.Track_ft.value_counts())
    # fiun
    #Sdog_results = dog_results[~dog_results['track_code']]
    ohe = OneHotEncoder(sparse_output=False)
    all_results = pd.read_feather(prev_full_details)
    print(len(ohe.fit_transform(all_results[['Track']])[0]))
    with open('encoder_new', 'wb') as f:
        pickle.dump(ohe, f)
    simple = False
    normalize = False

    simp = ''
    norm = ''

    # test = pd.read_pickle(prev_full_details)

    if simple:
        simp = '_simple'
    if normalize:
        norm = '_normed'

    x,stats_cols = featurecreations.generate_results_df_v2(prev_full_details, dog_results, 'na', betfair_sp_file, split_dist_file,mode="OO", prev_df="AHHHHH", ohe='encoder_new', simple=simple, normalize=normalize,v6=False)

    print(x)

    x['stats_cols'] = str([stats_cols])
    x['stats_cols'] = x['stats_cols'].astype('category')

    logging.info(f"Stats cols = {stats_cols}")
    logging.info(f"DB size = {x.shape}")

    x.reset_index().to_feather(f"./DATA/gru_inputs{simp}{norm}_kitchen_sink_new_not_simple.fth")
    

    # if simple:
    #     with open(f"./DATA/gru_inputs_new{simp}{norm}_test.npy", 'wb') as f:
    #         pickle.dump(x, f)
    # else:
    #     with open(f"./DATA/gru_inputs_new{simp}{norm}.npy", 'wb') as f:
    #         pickle.dump(x, f)

    # x.to_pickle(r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\DATA\gru_inputs_new_pir.npy")


