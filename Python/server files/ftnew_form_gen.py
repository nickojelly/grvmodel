import pandas as pd
import os
import betfairlightweight
from datetime import datetime
from datetime import timedelta
import pandas as pd
import fasttrack as ft
import importlib
import featurecreations

# Import libraries for logging in
from flumine import Flumine, clients
import logging 
import re
from ft_sec_key import SECKEY
from nltk.tokenize import regexp_tokenize
import sys

TRACK_RENAME =  {'Sandown (SAP)':'Sandown Park',
                    'Palmerston North':'Manawatu',
                    'Christchurch':'Addington',
                    'Auckland':'Manukau',
                    'Waikato':'Manukau',
                    'Wagga Wagga':'Wagga',
                    'Southland':'Ascot Park', 
                    'Richmond Straight':'Richmond',
                    'Wanganui':'Hatrick'}

def get_fasttrack_data(race_date):

    greys = ft.Fasttrack(SECKEY)
    today = race_date

    ahead3 = (datetime.today()+timedelta(days=1)).strftime('%Y-%m-%d')

    
    race_details, dog_results = greys.getBasicFormat(today)
    race_details = race_details[['@id', 'RaceNum', 'Distance', 'Date','RaceTime', 'RaceName', 'RaceGrade', 'Track', 'State']]
    race_details['RaceId'] = race_details['@id']
    dog_results = dog_results[['@id', 'RaceBox', 'DogName', 'RaceId', 'DogGrade']]
    dog_results['DogId'] = dog_results['@id']
    dog_results['box'] = dog_results['RaceBox']

    fasttrack_df = pd.merge(dog_results,race_details, how='left', on='RaceId')
    fasttrack_df.DogName = fasttrack_df.DogName.apply(lambda x: x.replace("'", "").replace(".", "").replace("Res", "").strip() if x else "NA")

    fasttrack_df['tab_track'] = fasttrack_df['Track']

    track_rename_dict = {'Sandown (SAP)':'Sandown Park',
                        'Palmerston North':'Manawatu',
                        'Christchurch':'Addington',
                        'Auckland':'Manukau',
                        'Waikato':'Manukau',
                        'Wagga Wagga':'Wagga',
                        'Southland':'Ascot Park', 
                        'Richmond Straight':'Richmond',
                        'Wanganui':'Hatrick'}
    
    fasttrack_df['tab_track'] = fasttrack_df.Track.apply(lambda x: TRACK_RENAME[x] if x in TRACK_RENAME.keys() else x)

    return fasttrack_df

def get_betfair_data():
    # Credentials to login and logging in 
    trading = betfairlightweight.APIClient('nickbarlow@live.com.au','un6/chxe!N!?adsp',app_key='JFWqJHqB4Akfi5hK')
    client = clients.BetfairClient(trading, interactive_login=True)

    # Login
    framework = Flumine(client=client)

    trading.login_interactive()

    racing_filter=betfairlightweight.filters.market_filter(
            event_type_ids=["4339"], # Greyhounds
            market_countries=["AU", "NZ"], # Australia
            market_type_codes=["WIN"], # Win Markets
        )

    results = trading.betting.list_events(racing_filter)

    for i in results:
        print(i.event.id, i.event.name, i.market_count)

    results = trading.betting.list_venues(racing_filter)
    venues = [v.venue for v in results]
    for i in results:
        print(i, i.venue,i.market_count)

    results_list = []

    for v in venues:
        racing_filter=betfairlightweight.filters.market_filter(
            event_type_ids=["4339"], # Greyhounds
            market_countries=["AU", "NZ"], # Australia
            market_type_codes=["WIN"], # Win Markets
            venues=[v]
        )

        results = trading.betting.list_market_catalogue(
                market_projection=[
                    "RUNNER_DESCRIPTION", 
                    "RUNNER_METADATA", 
                    "COMPETITION", 
                    "EVENT", 
                    "EVENT_TYPE", 
                    "MARKET_DESCRIPTION", 
                    "MARKET_START_TIME",
                ],
                filter=racing_filter,
                max_results=110,
            )

        results_list.extend(results)

    df_list = []

    box_change_list = []
    pattern1 = r'(?<=<br>Dog ).+?(?= starts)'
    pattern2 = r"(?<=\bbox no. )(\w+)"

    for i in results_list:
        # print(f"{i.market_id,i.market_name,i.market_start_time.hour, i.market_start_time.minute, i.event.venue,  i.description.market_type} ")
        market_id = i.market_id
        race_num = int(re.sub("[^0-9]", "", i.market_name.split(' ',1)[0] ))
        track =  i.event.venue
        dist = i.market_name.split(' ',2)[0]
        # print(race_num)
        if i.description.clarifications:
            # print(i.description.clarifications.replace("<br> Dog","<br>Dog"))
            box_change_list.append((track, race_num,i.description.clarifications.replace("<br> Dog","<br>Dog")))

        for dog in i.runners:
            # print(f"id = {dog.selection_id}, name = {dog.runner_name.split(' ',1)[1].upper()}")
            df_list.append([market_id, track, dist, race_num, dog.selection_id,dog.runner_name.split('.',1)[0].upper(), dog.runner_name.split(' ',1)[1].upper()])
    df = pd.DataFrame(data = df_list, columns=['market_id', 'tab_track', 'dist', 'race_num', 'runner_id','box_bf', 'runnner_name'])
    box_change_df = pd.DataFrame.from_records(box_change_list, columns = ['tab_track', 'RaceNum', 'string'])

    #box_change_df.drop(index=0, inplace=True)
    box_change_df = box_change_df[~(box_change_df['string']==' ')]
    # box_change_df.replace('Wagga','Wagga Wagga',inplace = True)
    # box_change_df.replace('Manawatu','Palmerston North',inplace = True)
    # box_change_df.replace('Manukau','Waikato',inplace = True)
    # box_change_df.replace('Addington','Christchurch',inplace = True)
    # box_change_df.replace('Hatrick','Wanganui',inplace = True)
    box_change_df = box_change_df[~(box_change_df['string']=='  ')]
    box_change_df['runner_name'] = box_change_df['string'].apply(lambda x: regexp_tokenize(x, pattern1)[0])
    box_change_df['runner_name'] = box_change_df['string'].apply(lambda x: regexp_tokenize(x, pattern1)[0])
    box_change_df['runner_name_2'] = box_change_df['string'].apply(lambda x: regexp_tokenize(x, pattern1)[-1])

    box_change_df['runner_number'] = box_change_df['runner_name'].apply(lambda x: x[:(x.find(" ") - 1)].upper())
    box_change_df['runner_name'] = box_change_df['runner_name'].apply(lambda x: x[(x.find(" ") + 1):].upper())
    box_change_df['RaceBox'] = pd.to_numeric(box_change_df['string'].apply(lambda x: regexp_tokenize(x, pattern2)[0]))

    return df,box_change_df

if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if 'v6' in sys.argv:
        v6='_v6'
    else:
        v6=''
    if 'tomorrow' in sys.argv:
        tomorrow=True
        race_date = (datetime.today()+timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        tomorrow=False
        race_date = datetime.today().strftime('%Y-%m-%d')
    print(f'{v6=}')

    today = datetime.today().strftime('%Y-%m-%d')

    fasttrack_df = get_fasttrack_data(race_date)

    betfair_df, box_change_df = get_betfair_data()

    tracks_FT = fasttrack_df.Track.unique()

    print(f"{tracks_FT=}")

    logging.basicConfig(filename='database_updater.log', encoding='utf-8', level=logging.DEBUG)
    
    logging.info(f'--- {today} --- \n')
    logging.info(f"Creating new inputs for model predictions for {race_date=}\n{sys.argv=}")

    races_bf_df = betfair_df[['tab_track','race_num', 'box_bf']].copy()
    races_bf_df.columns = ['tab_track', 'RaceNum', 'RaceBox']



    races_bf_df.RaceBox = pd.to_numeric(races_bf_df.RaceBox)


    fasttrack_df.RaceNum = pd.to_numeric(fasttrack_df.RaceNum)
    fasttrack_df.RaceBox = pd.to_numeric(fasttrack_df.RaceBox)
    fasttrack_df = fasttrack_df.reset_index(drop=True).set_index(['tab_track', 'RaceNum', 'RaceBox'])
    fasttrack_df = fasttrack_df.drop(index=list(box_change_df[['tab_track','RaceNum','RaceBox']].itertuples(index=False)), errors='ignore').reset_index()
    # fasttrack_df = fasttrack_df.merge(races_bf_df, on=['Track', 'RaceNum', 'RaceBox'])
    # merged = fasttrack_df.merge(how='left', right=box_change_df, right_on=['runner_name', 'RaceNum'], left_on=['DogName', 'RaceNum'])
    merged = fasttrack_df
    print(f'{merged=}')
    print(f'{fasttrack_df=}')
    merged['Box'] = fasttrack_df.apply(lambda x: x['RaceBox'], axis =1)
    merged['race_time']  =  pd.to_datetime(merged.apply(lambda x: f"{x['Date']} {x['RaceTime']}",axis=1))
    new_form = merged[merged['Box']<=8][['Track','tab_track','State','DogId','DogName','DogGrade', 'RaceId','Distance', 'Date', 'Box','race_time','RaceTime','RaceNum']].rename(columns={'Track_x':'Track', 'RaceNum':'race_num'})

    importlib.reload(featurecreations)
    if 'v6' in sys.argv:
        x,stats_cols = featurecreations.generate_prediction_dataframe_v2('results-df-merged-prices_cut.fth', new_form, simple=True,v6=True)
    else:
        x,stats_cols = featurecreations.generate_prediction_dataframe_v2('results-df-merged-prices_cut.fth', new_form, simple=True)
    print(f"input size = {len(x['stats'][0])}")

    x['stats_cols'] = str([stats_cols])
    x['stats_cols'] = x['stats_cols'].astype('category')

    x = x[x['raceid'].isin( new_form['RaceId'])]



    print(x['prev_race_date'])
    print(x.track_name.value_counts())
    print(today)
    x.reset_index(drop=True).to_feather(f"//root/grv_model/model_predictions/prediction_inputs/testing new outs simple {race_date}{v6}.fth")
    x.reset_index(drop=True).to_excel(f"//root/grv_model/model_predictions/prediction_inputs/testing new outs simple {race_date}{v6}.xlsx")

    today = datetime.today().strftime('%Y-%m-%d %H:%M:%')
    logging.info(today)
    logging.info(f"{race_date=}")
    logging.info(x.track_name.value_counts())
    logging.info(f"Number of races {x.shape}")

    # with open(f"testing new outs simple {today}.npy", 'wb') as f:
    #     x = (x, ['speed_avg_1', 'speed_max_1', 'split_speed_avg_1', 'split_speed_max_1', 'split_margin_avg_1', 'margin_avg_1', 'first_out_avg_1', 'post_change_avg_1', 'races_1', 'wins_1'])
    #     pickle.dump(x, f)
