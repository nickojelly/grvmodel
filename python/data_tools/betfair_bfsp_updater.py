from datetime import date, timedelta
import pickle
import requests
from tqdm import tqdm
import os
import pandas as pd
import os
import pickle
import time
import math
import concurrent.futures

def download_file(links):
    for link in links:
        try:
            # print(link)
            fname = f"./DATA/BetfairSPdata/{link[-12:]}"
            response = requests.get(link)
            print(response.status_code)

            if response.status_code == 429:
                time.sleep(120)

            if response:
                open(fname, "wb").write(response.content)
                print(fname)
        except Exception as e:
            print(e)
            time.sleep(5)

            fname = f"./DATA/BetfairSPdata/{link[-12:]}"
            response = requests.get(link)
            if response:
                open(fname, "wb").write(response.content)
                print(fname)

def download_betfair_files(start_date):
    end_date = date.today() + timedelta(days=1)  # perhaps date.now()
    end_date = date(2018, 5, 1) 

    num_workers = 6
    

    delta = end_date - start_date  # returns timedelta

    day_list = []
    for i in range(delta.days):
        day = start_date + timedelta(days=i)
        print(day.strftime('%d%m%y'))
        day_list.append(f"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin{day.strftime('%d%m%Y')}.csv")

    # Use a ThreadPoolExecutor to download the files in parallel
    chunk_size = math.ceil(len(day_list) / num_workers)
    chunks = [day_list[i:i + chunk_size] for i in range(0, len(day_list), chunk_size)]
    _process_jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for i,chunk in enumerate(chunks):
            _process_jobs.append(executor.submit(download_file, chunk, ))
    # for i in tqdm(day_list):
    #     download_file(i)

def update_bf_df():
    csv_file_dir = './DATA/BetfairSPdata'
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    csv_files = os.listdir(csv_file_dir)
    print(len(csv_files))
    # print(csv_files)
    # os.chdir(csv_file_dir)
    csv_columns = ['EVENT_ID', 'MENU_HINT', 'EVENT_NAME', 'EVENT_DT', 'SELECTION_ID',
        'SELECTION_NAME', 'WIN_LOSE', 'BSP', 'PPWAP', 'MORNINGWAP', 'PPMAX',
        'PPMIN', 'IPMAX', 'IPMIN', 'MORNINGTRADEDVOL', 'PPTRADEDVOL',
        'IPTRADEDVOL']

    df = pd.concat((pd.read_csv(f"{csv_file_dir}/{f}", on_bad_lines='skip',header=0, names = csv_columns) for f in csv_files), ignore_index=True)
    df['date'] = pd.to_datetime(df.EVENT_DT, dayfirst=True,format='mixed',errors='coerce')
    print(df.dtypes)
    print(df['date'].value_counts())

    df = df[df['BSP'].notnull()]
    df["dog"] = df["SELECTION_NAME"].astype(str).str[2:]
    df["loc"] = df['MENU_HINT'].str[0:3]
    df.shape
    dfAus = df[(df['MENU_HINT'].str.contains("(AUS)"))|(df['MENU_HINT'].str.contains("(NZL)"))].copy()
    dfAus.shape
    dfAus['date_bfsp'] = pd.to_datetime(dfAus.EVENT_DT, dayfirst=True,format='mixed',errors='coerce').dt.date
    # print(dfAus.max())
    dfAus = dfAus[['EVENT_ID', 'EVENT_DT', 'SELECTION_ID', 'BSP','dog','MENU_HINT']]
    # os.chdir(r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\Database Updater\DATA")

    dfAus.to_feather("./DATA/df-betfairSP.fth")
    # with open("./DATA/df-betfairSP.npy", "wb") as fp:   #Pickling
        
    #     pickle.dump(dfAus, fp)

if __name__ == "__main__":

    start_date = date(2015, 1, 1) 
    # download_betfair_files(start_date)

    update_bf_df()