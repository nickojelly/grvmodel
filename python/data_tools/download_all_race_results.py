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

    race_details, dog_results = greys.getRaceResults('2019-12-01', dt_end=today)
    # race_details = race_details[['@id', 'RaceNum', 'Distance', 'Date','RaceTime', 'RaceName', 'RaceGrade', 'Track']]
    race_details['RaceId'] = race_details['@id']
    #dog_results = dog_results[['@id', 'RaceBox', 'DogName', 'RaceId']]
    dog_results['DogId'] = dog_results['@id']
    #dog_results['box'] = dog_results['RaceBox']

    race_details.to_pickle('all results races.npy')
    dog_results.to_pickle('all results dogs.npy')
    # prediction_df = pd.merge(dog_results,race_details, how='left', on='RaceId')

    # x = featurecreations.generate_prediction_dataframe("results-df-merged-prices.npy", prediction_df)

    # x.to_pickle(f'prediction_input {today}.npy')