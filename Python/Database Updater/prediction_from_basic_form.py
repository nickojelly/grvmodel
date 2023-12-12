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

if __name__=="__main__":
    seckey = "50efd775-e988-4be3-924f-87631fabdc3f"
    greys = ft.Fasttrack(seckey)
    today = datetime.today().strftime('%Y-%m-%d')

    race_details, dog_results = greys.getBasicFormat(today, dt_end=today)
    race_details['RaceId'] = race_details['@id']
    dog_results['DogId'] = dog_results['@id']
    dog_results['box'] = dog_results['RaceBox']
    basic_form = dog_results.merge(race_details,how='left', on='RaceId')
    #basic_form = basic_form.rename(columns={'@id_x':'DogId','@id_y':'RaceId'})

    race_details.to_pickle(f'all basic form races{today}.npy')
    dog_results.to_pickle(f'all basic form dogs{today}.npy')

    prediction_df = pd.merge(dog_results,race_details, how='left', on='RaceId')

    x = featurecreations.generate_prediction_dataframe("results-df-merged-prices.npy", basic_form, 'encoder')
    
    x.to_pickle(f'prediction_input {today}.npy')