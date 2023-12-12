import pickle
import pandas as pd
import os
import torch

import betfairlightweight
from betfairlightweight import filters
from datetime import datetime, timedelta
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
pd.set_option('display.max_rows', 100)

if __name__=="__main__":
    seckey = "50efd775-e988-4be3-924f-87631fabdc3f"
    greys = ft.Fasttrack(seckey)
    today = (datetime.today()+timedelta(days=1)).strftime('%Y-%m-%d')
    lastweek = (datetime.today()-timedelta(days=7)).strftime('%Y-%m-%d')

    race_details, dog_results = greys.getBasicFormat(lastweek, dt_end=today)
    race_details['RaceId'] = race_details['@id']
    dog_results['DogId'] = dog_results['@id']
    dog_results['box'] = dog_results['RaceBox']
    basic_form = dog_results.merge(race_details,how='left', on='RaceId')[['@id_x', '@id_y', 'DogGrade']]
    basic_form = basic_form.rename(columns={'@id_x':'DogId','@id_y':'RaceId'})

    # race_details.to_pickle(f'all basic form races{today}.npy')
    # dog_results.to_pickle(f'all basic form dogs{today}.npy')
    # print(os.getcwd())
    # old_race_basic_form = pd.read_pickle(r'./DATA/basic form/all basic form races.npy')
    # old_dog_basic_form = pd.read_pickle(r'./DATA/basic form/all basic form dogs.npy')
    old_merge = pd.read_pickle(r'./DATA/basic form/all basic form merged.npy')
    print(old_merge.columns)
    old_merge.to_pickle('all basic form merged.npy')
    all_races = pd.concat([old_merge,basic_form])
    all_races.drop_duplicates
    all_races.to_pickle(r'./DATA/basic form/all basic form merged.npy')
    #

    #x = featurecreations.generate_prediction_dataframe("results-df-merged-prices.npy", basic_form)

    #x.to_pickle(f'prediction_input {today}.npy')