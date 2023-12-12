from datetime import date, timedelta
import pickle
import requests
from tqdm import tqdm
import os
import pandas as pd
import os
import pickle
import time

def download_betfair_files(start_date):

    end_date = date.today() + timedelta(days=1)  # perhaps date.now()

    delta = end_date - start_date  # returns timedelta

    day_list = []
    for i in range(delta.days):
        day = start_date + timedelta(days=i)
        print(day.strftime('%d%m%y'))
        day_list.append(f"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin{day.strftime('%d%m%Y')}.csv")
    day_list
    # os.chdir(r'./DATA/BetfairSPdata')
    for link in tqdm(day_list):
        try:
            fname = f"./DATA/BetfairSPdata/{link[-12:]}"
            response = requests.get(link)
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

def update_bf_df():
    csv_file_dir = './DATA/BetfairSPdata'
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    csv_files = os.listdir(csv_file_dir)
    print(len(csv_files))
    # os.chdir(csv_file_dir)
    csv_march = [f for f in csv_files if '042023.csv' in f]
    df = pd.concat((pd.read_csv(f"{csv_file_dir}/{f}", on_bad_lines='skip') for f in csv_files), ignore_index=True)
    df['date'] = pd.to_datetime(df.EVENT_DT, dayfirst=True).dt.date

    df = df[df['BSP'].notnull()]
    df["dog"] = df["SELECTION_NAME"].astype(str).str[2:]
    df["loc"] = df['MENU_HINT'].str[0:3]
    df.shape
    dfAus = df[(df['MENU_HINT'].str.contains("(AUS)"))|(df['MENU_HINT'].str.contains("(NZL)"))]
    dfAus.shape
    dfAus = dfAus[['EVENT_ID', 'EVENT_DT', 'SELECTION_ID', 'BSP','dog','MENU_HINT']]
    # os.chdir(r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\Database Updater\DATA")
    dfAus.to_pickle("./DATA/df-betfairSP.npy")
    dfAus.to_feather("./DATA/df-betfairSP.fth")
    # with open("./DATA/df-betfairSP.npy", "wb") as fp:   #Pickling
        
    #     pickle.dump(dfAus, fp)

if __name__ == "__main__":

    start_date = date(2023, 9, 8) 
    download_betfair_files(start_date)

    update_bf_df()