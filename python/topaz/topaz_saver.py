import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from topaz import TopazAPI
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
import requests
import os
import concurrent.futures
import math
api_key = '51066aa1-85af-4f6f-aaa5-08b0f3133af5' #Insert your API key 
topaz_api = TopazAPI(api_key)


def topaz_race_runs_threaded(chunk,topaz_api:TopazAPI,progress):
    race_runs = []
    race_results = []
    errors = []
    for race_id in chunk:
        try:
            race_run = topaz_api.get_race_runs(race_id=race_id)
            race_runs.append(race_run)
            time.sleep(0.3)
            race_result_json = topaz_api.get_race_result(race_id = race_id)
            try:
                race_run.to_feather(f"race_runs/{race_id}_run.fth")
                race_result_df = pd.DataFrame.from_dict([race_result_json])
                race_result_df.to_feather(f"results/{race_id}_results.fth")
            except Exception as e:
                print(e)

        except requests.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
            if http_err.response.status_code == 429:

                time.sleep(120)
            errors.append(race_id)
            pass
        progress.update()

    return race_runs,race_results,errors

def topaz_race_run_getter(race_id_list,topaz_api:TopazAPI):

    print(f"Fetching data for  {len(race_id_list)}")

    num_workers = 6
    chunk_size = math.ceil(len(race_id_list) / num_workers)

    chunks = [race_id_list[i:i + chunk_size] for i in range(0, len(race_id_list), chunk_size)]
    
    print(chunks)
    print(len(chunks))
    _process_jobs = []
    bars = []
    race_runs = []
    results = []
    errors = []
    for i in range(num_workers):
        bars.append(tqdm(total=len(chunks[i]), position=i)) 
        # time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:


        for i,chunk in enumerate(chunks):
            _process_jobs.append(executor.submit(topaz_race_runs_threaded, chunk, topaz_api, bars[i]))

        # results = []
        for job in concurrent.futures.as_completed(_process_jobs):
            race_run,result_json,error = job.result()
            race_runs.extend(race_run)
            errors.extend(error)
            results.extend(result_json)

    

    # results = []
    print(errors)
    return race_runs,results,errors

if __name__ == '__main__':

    api_key = '51066aa1-85af-4f6f-aaa5-08b0f3133af5' #Insert your API key 
    topaz_api = TopazAPI(api_key)

    all_races_df = pd.read_csv('all_races_topas.csv', header=0)
    i = 0
    race_ids = list(all_races_df['raceId'].unique())

    for i in range(0,len(race_ids),1000):
        subset_ids = race_ids[i:min(len(race_ids),i+1000)]
        race_runs,results,errors = topaz_race_run_getter(subset_ids,topaz_api)

        results_df = pd.DataFrame.from_dict(results)
        all_race_runs = pd.concat(race_runs,ignore_index=True).reset_index(drop=True)
        all_race_runs.to_feather(f'race_runs/{i}_topaz_race_runs.fth')
        results_df.to_feather(f"results/{i}_topaz_results.fth") 