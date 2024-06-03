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
            race_result_json,response = topaz_api.get_race_result(race_id = race_id)
            rate_lim_left = int(response.headers['ratelimit-remaining'])
            reset_time = int(response.headers['ratelimit-reset'])
            if rate_lim_left < 5:
                time.sleep(reset_time+1)
            try:
                race_run = pd.DataFrame(race_result_json['runs'])
                split_times = pd.DataFrame(race_result_json['splitTimes'])
                if len(split_times) != 0:

                    split_times_1 = split_times[split_times['splitTimeMarker'] == 1][['runId','time','position','splitMargin']]
                    split_time_2 = split_times[split_times['splitTimeMarker'] == 2][['runId','time','position','splitMargin']]
                    split_times = split_times_1.merge(split_time_2, on='runId',suffixes=('_1','_2'),how='left')
                    race_run = race_run.merge(split_times, on='runId', how='left')  
                    # race_result_df = pd.DataFrame.from_dict([race_result_json])
                    #race_run.to_feather(f"results/{race_id}_results.fth")
                    
                else:
                    # print(f"No split time for race: {race_id}")
                    #race_run.to_feather(f"results/{race_id}_results.fth")
                    pass
                race_runs.append(race_run)
            except Exception as e:
                print(e)
        except requests.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')

            if http_err.response.status_code == 429:

                time.sleep(30)
            errors.append(race_id)
            pass
        except Exception as e:
            print(f'Other error occurred: {e}')
            errors.append(race_id)
        progress.update()

    return race_runs,errors

def topaz_race_run_getter(race_id_list,topaz_api:TopazAPI):

    print(f"Fetching data for  {len(race_id_list)}")

    num_workers = 2
    chunk_size = math.ceil(len(race_id_list) / num_workers)

    chunks = [race_id_list[i:i + chunk_size] for i in range(0, len(race_id_list), chunk_size)]
    
    print(chunks)
    print(len(chunks))
    _process_jobs = []
    bars = []
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
            race_runs,error = job.result()
            errors.extend(error)
            results.extend(race_runs)

    

    # results = []
    print(errors)
    return results,errors

def generate_date_range(start_date, end_date):
    start_date = start_date
    end_date = end_date

    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=31)

    return date_list

def topaz_races_threaded(buckets, topaz_api, progress):
    all_races = []
    # print(f"{buckets=}")
    errors = []
    for bucket in buckets:
        start_date, end_date, state = bucket
        # print(bucket)
        try:
            races = topaz_api.get_races(from_date=start_date, to_date=end_date, owning_authority_code=state)
            races['state'] = state
            all_races.append(races)
        except requests.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
            errors.append(bucket)
            pass
        # time.sleep(2)
        progress.update()
    return all_races,errors

def get_topaz_races(start_date, end_date, states, topaz_api:TopazAPI):
    date_range = generate_date_range(start_date, end_date)
    starts = date_range[:-1]
    ends = date_range[1:]
    date_range_states = [(start, end, state) for start, end in zip(starts, ends) for state in states]

    print(f"Created {len(date_range_states)} date ranges for {len(states)} states")

    num_workers = min(6, len(date_range_states))  # Adjust this value based on your system's capabilities
    chunk_size = math.ceil(len(date_range_states) / num_workers)

    chunks = [date_range_states[i:i + chunk_size] for i in range(0, len(date_range_states), chunk_size)]
    
    print(chunks)
    print(len(chunks))
    _process_jobs = []
    bars = []
    results = []
    errors = []
    for i in range(num_workers):
        bars.append(tqdm(total=len(chunks[i]), position=i)) 
        # time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:


        for i,chunk in enumerate(chunks):
            _process_jobs.append(executor.submit(topaz_races_threaded, chunk, topaz_api, bars[i]))

        # results = []
        for job in concurrent.futures.as_completed(_process_jobs):
            result,error = job.result()
            errors.extend(error)
            results.append(result)

    

    # results = []
    print(errors)
    return results

if __name__ == '__main__':

    api_key = '51066aa1-85af-4f6f-aaa5-08b0f3133af5' #Insert your API key 
    topaz_api = TopazAPI(api_key)

    states = ['NZ', "NSW", "VIC", "SA"]
    start_date = datetime(2024,5,14)
    end_date = (datetime.today() + timedelta(days=31))

    output = get_topaz_races(start_date, end_date, states, topaz_api)
    output_flat = [item for sublist in output for item in sublist]
    all_races_df = pd.concat(output_flat,ignore_index=True).reset_index(drop=True)
    all_races_df.to_csv('all_races_topas_NEW.csv',index=False)

    all_races_df = pd.read_csv('all_races_topas_NEW.csv', header=0)
    i = 0
    race_ids = list(all_races_df['raceId'].unique())
    #subset_ids = race_ids[250_000:len(race_ids)]
    #i = 250_000
    # for i in range(250_000,len(race_ids),10_000):
    #     subset_ids = race_ids[i:min(len(race_ids),i+10000)]
    race_runs,errors = topaz_race_run_getter( race_ids,topaz_api)

    # results_df = pd.DataFrame.from_dict(results)
    all_race_runs = pd.concat(race_runs,ignore_index=True).reset_index(drop=True)
    all_race_runs.to_feather(f'data_complete/NEW3_topaz_race_runs_w_split.fth')
    all_race_runs.to_csv(f'race_runs/_topaz_race_runs_w_split.csv')
    # results_df.to_feather(f"results/{i}_topaz_results.fth")