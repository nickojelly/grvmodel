import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import OneHotEncoder
import pickle
import tqdm
from operator import itemgetter
import math

TRACKS = pd.Series(data=['Mandurah', 'Hobart', 'Casino', 'Geelong', 'Mount Gambier',
       'Christchurch (NZ)', 'Ballarat', 'Sandown Park', 'Shepparton',
       'Waikato (NZ)', 'Gunnedah', 'Albion Park', 'Dapto', 'Angle Park',
       'Palmerston Nth (NZ)', 'Meadows (MEP)', 'Cannington', 'Gawler',
       'Capalaba', 'Bendigo', 'Warrnambool', 'Traralgon', 'Richmond',
       'Taree', 'Wentworth Park', 'Rockhampton', 'Murray Bridge (MBS)',
       'Healesville', 'Ipswich', 'Grafton', 'Warragul', 'Horsham',
       'Bulli', 'Temora', 'Townsville', 'Launceston', 'Maitland',
       'Auckland (NZ)', 'Nowra', 'Dubbo', 'Sale', 'Darwin', 'Broken Hill',
       'The Gardens', 'Moree', 'Lithgow', 'The Meadows', 'Wagga',
       'Murray Bridge (MBR)', 'Goulburn', 'Gosford', 'Southland (NZ)',
       'Richmond (RIS)', 'Young', 'Sandown (SAP)', 'Tamworth', 'Northam',
       'Bathurst', 'Potts Park', 'Coonamble', 'Otago (NZ)', 'Kempsey',
       'Bundaberg', 'Canberra', 'Devonport', 'Lismore', 'Cranbourne',
       'Wanganui (NZ)', 'Wauchope', 'Muswellbrook', None, 'Tokoroa (NZ)',
       'Coonabarabran', 'Taranaki (NZ)'], name='Tracks')

def track_id_gen(dist, track):
    if track == None:
        return "NA"
    if dist == None:
        return "NA"
    return track[0:4] + "-" + str(int(dist))

def round_down(num, divisor):
    return num - (num%divisor)

def generate_prev_race(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    original_cols = df_in.columns
    df[f'prev_race'] = df_g['RaceId'].shift(1).fillna(-1)
    df[f'prev_race_date'] = df_g['dateF1'].shift(1).fillna(-1)
    df[f'prev_race_track'] = df_g['Track'].shift(1).fillna(-1)
    df[f'prev_race_state'] = df_g['State'].shift(1).fillna(-1)
    return(df)

def generate_prev_weight(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    df[f'prev_weight'] = df_g['weight'].shift(1).fillna(-1)
    return(df)

def generate_prediction_dataframe_v2(prev_results_file:str, prediction_df, ohe='encoder_new',simple=False):
    #prev_results_file = "results-df-merged-prices.npy"

    form = pd.read_pickle(prev_results_file)

    # prev_full_details = r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\DATA\full_race_details_ft.csv"
    # prev_full_details = pd.read_csv(prev_full_details)
    ohe = pickle.load(open(ohe, 'rb'))

    pred_track_OHE = ohe.transform(prediction_df[["Track"]])

    prediction_df["tracksOnehot"] = pred_track_OHE.tolist()
    prediction_df["dist_x"] = prediction_df["Distance"].astype(str).str[:-1].astype(float)
    prediction_df['dist_round'] = prediction_df.dist_x.apply(lambda x: round_down(x, 50))
    prediction_df["date"] =  pd.to_datetime(prediction_df["Date"], format="%d %b %y").dt.date
    prediction_df['DogGrade'] = pd.to_numeric(prediction_df['DogGrade'],errors='coerce').fillna(8)
    prediction_df['box'] = pd.to_numeric(prediction_df['Box'],errors='coerce').fillna(8)
    race_pred_forms = prediction_df.sort_values("date", ascending=False).groupby(["RaceId"], sort=False)
    form["date"] =  pd.to_datetime(form["date"], format="%d %b %y").dt.date

    form = pd.concat([form, prediction_df])



    
    # form = form[form['DogId'].isin(prediction_df['DogId'])]

    form = form.reset_index(drop=True)
    dog_form = form.groupby(['DogId'], sort=False, as_index=False)
    dist_form = form.groupby(['DogId','dist_round'], sort=False, as_index=False)
    box_form = form.groupby(['DogId','box'], sort=False, as_index=False)
    track_box_form = form.groupby(['DogId','box','Track'], sort=False, as_index=False)
    track_dist_form = form.groupby(['DogId','dist_x','Track'], sort=False, as_index=False)
    form['win'] = form['place'].apply(lambda x: 1 if x ==1 else 0)
    form['count'] = 1


    form = form.reset_index(drop=True)
    dog_form = form.groupby(['DogId'], sort=False, as_index=False)
    dist_form = form.groupby(['DogId','dist_round'], sort=False, as_index=False)
    box_form = form.groupby(['DogId','box'], sort=False, as_index=False)
    track_box_form = form.groupby(['DogId','box','Track'], sort=False, as_index=False)
    track_dist_form = form.groupby(['DogId','dist_x','Track'], sort=False, as_index=False)

    form = generate_prev_weight(form, dog_form)
    if simple:
        form_stats = generate_dog_stats(form, dog_form,rolling_window=1)
        print("fin 1")
    else:
        form = generate_prev_weight(form, dog_form)
        form_stats = generate_dog_stats(form, dog_form,rolling_window=30)
        print("fin 30")
        form_stats = generate_dog_stats(form_stats, dog_form,rolling_window=5)
        print("fin 5")
        form_stats = generate_dog_stats(form_stats, dog_form,rolling_window=1)
        print("fin 1")
        form_stats = generate_dog_stats(form_stats, dist_form,rolling_window=10,factor='dist')
        print("fin 10_dist")
        form_stats = generate_dog_stats(form_stats, box_form,rolling_window=10,factor='box')
        print("fin 10 box")
        form_stats = generate_dog_stats(form_stats, track_box_form,rolling_window=10,factor='track_box')
        print("fin 10 track_box")
        form_stats = generate_dog_stats(form_stats, track_dist_form,rolling_window=10,factor='track_dist')
        print("fin 10 track_dist")  
    

    #Status columns hold all col names of generated stats
    stats_cols = [x for x in form_stats.columns if x not in form.columns]
    print(stats_cols)
    form_only_stats = form_stats[['box', 'prev_weight',  'DogGrade']+stats_cols].fillna(-1.0).values
    stats = pd.Series(form_only_stats.tolist())

    form = form_stats.drop(columns=stats_cols)
    form['stats'] = stats

    form = generate_prev_race(form, dog_form)

    form = form[['DogId', 'DogName', 'RaceId','race_num','RaceGrade', 'date', 'race_time','tracksOnehot','Track','State','track_code','dist_x', 'BSP','RunTime','prev_race','prev_race_date','prev_race_track','Place','margin','stats']]
    form.columns=['dogid', 'dog_name', 'raceid','race_num','race_grade', 'date','race_time', 'trackOHE','track_name', 'state', 'track_code','dist','bfSP','runtime','prev_race', 'prev_race_date','prev_race_track', 'place', 'margin','stats']

    return form


def generate_results_df_v2(previous_results_file, new_FT : pd.DataFrame ,new_FT_race_data : pd.DataFrame, betfair_sp_file :str, split_dist:str, mode="all", prev_df=None, ohe='encoder_new', simple=False):
    ohe = pickle.load(open(ohe, 'rb'))

    previous_results = pd.read_pickle(previous_results_file)
    previous_results = dd.
    # dog_results = previous_results[['DogId', 'Place', 'DogName', 'Box', 'Rug', 'Weight', 'StartPrice',
    #    'Handicap', 'Margin1', 'Margin2', 'PIR', 'Checks', 'Comments',
    #    'SplitMargin', 'RunTime', 'Prizemoney', 'RaceId', 'TrainerId',
    #    'TrainerName', 'RaceNum', 'RaceName', 'RaceTime', 'Distance',
    #    'RaceGrade', 'Track_ft', 'Track', 'State', 'track_code', 'date',
    #    'DogGrade']] #pd.concat([previous_results, dog_results])
    # print(f"Previous Results {len(dog_results), dog_results.shape}")

    

    dog_results = new_FT.copy()
    dog_results['check'] = dog_results['Place'].apply(lambda x: 1 if isinstance(x, str) else 0)
    dog_results = dog_results[dog_results['check']==1]  
    dog_results['Place'] = dog_results['Place'].apply(lambda x: x[0] if isinstance(x, str) else x)
    dog_results['Place'] = pd.to_numeric(dog_results['Place'], errors='coerce')
    print(len(dog_results))
    dog_results = dog_results.dropna(subset=['Place'])
    print(len(dog_results))
    dog_results.shape  
    dog_results['SplitTimes']= dog_results.SplitMargin.astype(float)
    dog_results['minMargin'] = dog_results.groupby('RaceId')['SplitTimes'].transform('min')
    dog_results['SplitMargin'] = dog_results.SplitTimes-dog_results.minMargin
    dog_results['Margin1'] = pd.to_numeric(dog_results['Margin1'], errors='coerce').fillna(0)
    dog_results['StartPrice'] = dog_results['StartPrice'].fillna(0)
    dog_results['DogGrade'] = pd.to_numeric(dog_results['DogGrade'], errors='coerce').fillna(8)

    dog_results['PIR_adj'] = dog_results.PIR.str.replace('[^0-9]', '').fillna(0)
    dog_results['start_pos'] =  dog_results['PIR_adj'].apply(lambda x: x[0] if x else 8)
    dog_results['last_pos'] =  dog_results['PIR_adj'].apply(lambda x: x[-1] if x else 8)
    dog_results['first_out'] = dog_results['start_pos'].apply(lambda x: 1 if x==1 else 0)
    dog_results['pos_change'] = pd.to_numeric(dog_results['start_pos']) - pd.to_numeric(dog_results['last_pos'])

    # full_details = pd.merge(dog_results,race_details, how='left', on='RaceId')
    
    betfair_df = pickle.load(open(betfair_sp_file, 'rb'))
    betfair_df['dateF'] = pd.to_datetime(betfair_df.EVENT_DT, dayfirst=True).dt.date
    betfair_df['dog_name'] = betfair_df.dog.str[1:].str.upper()

    resultsdf = dog_results
    print(len(resultsdf))

    # new_results_track_OHE = ohe.transform(resultsdf[["Track"]])
    # resultsdf["tracksOnehot"] =  new_results_track_OHE.tolist()
    print(resultsdf.columns)


    split_distances = pd.read_csv(split_dist)
    resultsdf["dist"] = resultsdf["Distance"].astype(str).str[:-1].astype(float)
    resultsdf = resultsdf[resultsdf['RunTime'].notnull()]
    resultsdf['run_time'] = pd.to_numeric(resultsdf['RunTime'])
    resultsdf['split_margins'] = resultsdf.SplitMargin.astype(float)

    
    print(resultsdf.columns)
    resultsdf['place'] = resultsdf.Place.astype(float)
    track_ids = resultsdf.apply(lambda s: track_id_gen(s["dist"], s["Track"]), axis=1 )
    print(len(track_ids))
    print(resultsdf.shape)
    resultsdf["track_id"] = resultsdf.apply(lambda s: track_id_gen(s["dist"], s["Track"]), axis=1 )
    print(resultsdf.columns)
    resultsdf_merged = pd.merge(resultsdf, split_distances, on=["track_id", "Track"], how='left')
    resultsdf_merged = resultsdf_merged[resultsdf_merged['RunTime'].notnull()]
    print(resultsdf_merged.columns)
    resultsdf_merged['split_dist_estim'] = resultsdf_merged['split_dist_estim'].fillna(100)
    resultsdf_merged["dateF1"] = pd.to_datetime(resultsdf_merged["date"], format="%d %b %y")
    resultsdf_merged["StartPrice_num"] = pd.to_numeric(resultsdf_merged['StartPrice'].str.replace('\$','').replace('F',''), errors='coerce').fillna(0)
    resultsdf_merged["speed"] = pd.to_numeric(resultsdf_merged["dist_x"])/pd.to_numeric(resultsdf_merged["RunTime"])

    resultsdf_merged["split_speed"] = pd.to_numeric(resultsdf_merged["split_dist_estim"])/pd.to_numeric(resultsdf_merged["SplitTimes"])

    resultsdf_merged["box"] = pd.to_numeric(resultsdf_merged["Box"])
    resultsdf_merged["margin"] = pd.to_numeric(resultsdf_merged["Margin1"])
    resultsdf_merged["weight"] = pd.to_numeric(resultsdf_merged["Weight"])
    resultsdf_merged['split_margins'] = pd.to_numeric(resultsdf_merged.SplitMargin)
    resultsdf_merged["dateF"] = pd.to_datetime(resultsdf_merged["date"], format="%d %b %y").dt.date
    resultsdf_merged = resultsdf_merged[resultsdf_merged['DogName'].notna()]
    resultsdf_merged['dog_name'] = resultsdf_merged['DogName'].apply(lambda x: x.replace("'", "").replace(".", "").replace("Res", "").strip())

    resultsdf_merged['dist_round'] = resultsdf_merged['dist_x'].apply(lambda x: round_down(x, 50))

    resultsdf_merged = pd.merge(
        resultsdf_merged, betfair_df, how="left", on=["dateF", "dog_name"]
    )

    form = resultsdf_merged.sort_values("dateF", ascending=True)
    form.loc[form['place']==1, 'margin']=0

    previous_results = pd.read_pickle(previous_results_file)
    all_results = pd.concat([previous_results, resultsdf_merged])
    # all_results = pd.concat([previous_results, resultsdf_merged])

    all_results['race_time'] = pd.to_datetime(all_results['RaceTime'])

    all_results =   all_results.drop_duplicates( subset=['RaceId', 'DogId'], keep='last')



    new_results_track_OHE = ohe.transform(all_results[["Track"]])
    all_results["tracksOnehot"] =  new_results_track_OHE.tolist()

    form = all_results.sort_values("dateF", ascending=True)

    form.loc[form['place']==1, 'margin']=0

    

    form['win'] = form['place'].apply(lambda x: 1 if x ==1 else 0)
    form['count'] = 1


    with open("results-df-merged-prices.npy", "wb") as fp:   #Pickling
    
        pickle.dump(all_results, fp)

    # form = all_results.sort_values(['dateF'], ascending=False)
    # dog_forms = form.groupby(["DogId"])

    print("starting stats rolling window")


    #NEW Groups for functions
    form = form.reset_index(drop=True)
    dog_form = form.groupby(['DogId'], sort=False, as_index=False)
    dist_form = form.groupby(['DogId','dist_round'], sort=False, as_index=False)
    box_form = form.groupby(['DogId','box'], sort=False, as_index=False)
    track_box_form = form.groupby(['DogId','box','Track'], sort=False, as_index=False)
    track_dist_form = form.groupby(['DogId','dist_x','Track'], sort=False, as_index=False)

    if simple:
        form_stats = generate_dog_stats(form, dog_form,rolling_window=1)
        print("fin 1")
    else:
        form = generate_prev_weight(form, dog_form)
        form_stats = generate_dog_stats(form, dog_form,rolling_window=30)
        print("fin 30")
        form_stats = generate_dog_stats(form_stats, dog_form,rolling_window=5)
        print("fin 5")
        form_stats = generate_dog_stats(form_stats, dog_form,rolling_window=1)
        print("fin 1")
        form_stats = generate_dog_stats(form_stats, dist_form,rolling_window=10,factor='dist')
        print("fin 10_dist")
        form_stats = generate_dog_stats(form_stats, box_form,rolling_window=10,factor='box')
        print("fin 10 box")
        form_stats = generate_dog_stats(form_stats, track_box_form,rolling_window=10,factor='track_box')
        print("fin 10 track_box")
        form_stats = generate_dog_stats(form_stats, track_dist_form,rolling_window=10,factor='track_dist')
        print("fin 10 track_dist")  

    #Status columns hold all col names of generated stats
    stats_cols = [x for x in form_stats.columns if x not in form.columns]
    print(f"STATS COLS {stats_cols}")
    form_only_stats = form_stats[['box', 'weight',  'DogGrade']+stats_cols].fillna(-1.0).values
    stats = pd.Series(form_only_stats.tolist())
    # form_stats.to_csv('form data for R analasis.csv')
    form = form_stats.drop(columns=stats_cols)
    form['stats'] = stats

    form = generate_prev_race(form, dog_form)

    form = form[['DogId', 'DogName', 'RaceId', 'RaceGrade', 'dateF','race_time','RaceNum','tracksOnehot','Track','State','track_code','dist_x', 'BSP','RunTime','prev_race','prev_race_date','prev_race_track','prev_race_state','Place','margin','stats']]
    form.columns=['dogid', 'dog_name', 'raceid', 'race_grade', 'date','race_time','race_num','trackOHE','track_name', 'state', 'track_code','dist','bfSP','runtime','prev_race', 'prev_race_date','prev_race_track','prev_race_state', 'place', 'margin','stats']

    return (form,stats_cols)


def generate_dog_stats_simple(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    df[f'speed_avg{factor}_{rolling_window}'] = df_g['speed'].rolling(rolling_window, min_periods=1, closed='left').mean()['speed']
    df[f'split_speed_avg{factor}_{rolling_window}'] = df_g['split_speed'].rolling(rolling_window, min_periods=1, closed='left').mean()['split_speed']
    df[f'split_margin_avg{factor}_{rolling_window}'] = df_g['split_margins'].rolling(rolling_window, min_periods=1, closed='left').mean()['split_margins']
    df[f'margin_avg{factor}_{rolling_window}'] = df_g['margin'].rolling(rolling_window, min_periods=1, closed='left').mean()['margin']
    df[f'first_out_avg{factor}_{rolling_window}'] = df_g['first_out'].rolling(rolling_window, min_periods=1, closed='left').mean()['first_out']
    df[f'post_change_avg{factor}_{rolling_window}'] = df_g['pos_change'].rolling(rolling_window, min_periods=1, closed='left').mean()['pos_change']
    df[f'races{factor}_{rolling_window}'] = df_g.cumcount().reset_index(drop=True)
    df[f'wins{factor}_{rolling_window}'] = df_g['win'].rolling(1000, min_periods=1, closed='left').sum()['win']
    df[f'weight_{factor}'] = df_g['weight'].rolling(rolling_window, min_periods=1, closed='left').mean()['weight']
    df[f'min_time_{factor}'] = df_g["RunTime"].rolling(rolling_window, min_periods=1, closed='left').min()["RunTime"]
    df[f'min_split_time_{factor}'] = df_g["SplitTimes"].rolling(rolling_window, min_periods=1, closed='left').min()["SplitTimes"]
    return(df)

def generate_dog_stats(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    df[f'speed_avg{factor}_{rolling_window}'] = df_g['speed'].rolling(rolling_window, min_periods=1, closed='left').mean()['speed']
    df[f'speed_max{factor}_{rolling_window}'] = df_g['speed'].rolling(rolling_window, min_periods=1, closed='left').max()['speed']

    df[f'split_speed_avg{factor}_{rolling_window}'] = df_g['split_speed'].rolling(rolling_window, min_periods=1, closed='left').mean()['split_speed']
    df[f'split_speed_max{factor}_{rolling_window}'] = df_g['split_speed'].rolling(rolling_window, min_periods=1, closed='left').max()['split_speed']

    df[f'split_margin_avg{factor}_{rolling_window}'] = df_g['split_margins'].rolling(rolling_window, min_periods=1, closed='left').mean()['split_margins']
    df[f'margin_avg{factor}_{rolling_window}'] = df_g['margin'].rolling(rolling_window, min_periods=1, closed='left').mean()['margin']

    df[f'first_out_avg{factor}_{rolling_window}'] = df_g['first_out'].rolling(rolling_window, min_periods=1, closed='left').mean()['first_out']
    df[f'post_change_avg{factor}_{rolling_window}'] = df_g['pos_change'].rolling(rolling_window, min_periods=1, closed='left').mean()['pos_change']

    df[f'races{factor}_{rolling_window}'] = df_g.cumcount().reset_index(drop=True)
    df[f'wins{factor}_{rolling_window}'] = df_g['win'].rolling(1000, min_periods=1, closed='left').sum()['win']
    if factor =='all':
        df[f'weight_{factor}'] = df_g['weight'].rolling(rolling_window, min_periods=1, closed='left').mean()['weight']
    if factor =="track_dist":
        df[f'min_time_{factor}'] = df_g["RunTime"].rolling(rolling_window, min_periods=1, closed='left').min()["RunTime"]
        df[f'mean_time_{factor}'] = df_g["RunTime"].rolling(rolling_window, min_periods=1, closed='left').mean()["RunTime"]
        df[f'min_split_time_{factor}'] = df_g["SplitTimes"].rolling(rolling_window, min_periods=1, closed='left').min()["SplitTimes"]
        df[f'mean_split_time_{factor}'] = df_g["SplitTimes"].rolling(rolling_window, min_periods=1, closed='left').mean()["SplitTimes"]
    return(df)

# def generate_dog_pred_stats(df_in, df_g, rolling_window=10, factor=''):
#     df = df_in.copy()
#     df[f'speed_avg{factor}_{rolling_window}'] = df_g['speed'].rolling(rolling_window, min_periods=1, closed='left').mean()['speed']
#     df[f'speed_max{factor}_{rolling_window}'] = df_g['speed'].rolling(rolling_window, min_periods=1, closed='left').max()['speed']
#     df[f'split_speed_avg{factor}_{rolling_window}'] = df_g['split_speed'].rolling(rolling_window, min_periods=1, closed='left').mean()['split_speed']
#     df[f'split_speed_max{factor}_{rolling_window}'] = df_g['split_speed'].rolling(rolling_window, min_periods=1, closed='left').max()['split_speed']
#     df[f'split_margin_avg{factor}_{rolling_window}'] = df_g['split_margins'].rolling(rolling_window, min_periods=1, closed='left').mean()['split_margins']
#     df[f'margin_avg{factor}_{rolling_window}'] = df_g['margin'].rolling(rolling_window, min_periods=1, closed='left').mean()['margin']

#     df[f'first_out_avg{factor}_{rolling_window}'] = df_g['first_out'].rolling(rolling_window, min_periods=1, closed='left').mean()['first_out']
#     df[f'post_change_avg{factor}_{rolling_window}'] = df_g['pos_change'].rolling(rolling_window, min_periods=1, closed='left').mean()['pos_change']

#     df[f'races{factor}_{rolling_window}'] = df_g.cumcount().reset_index(drop=True)
#     df[f'wins{factor}_{rolling_window}'] = df_g['win'].rolling(1000, min_periods=1, closed='left').sum()['win']
#     if factor =='all':
#         df[f'weight_{factor}'] = df_g['weight'].rolling(rolling_window, min_periods=1, closed='left').mean()['weight']
#     if factor =="track_dist":
#         df[f'min_time_{factor}'] = df_g["RunTime"].rolling(rolling_window, min_periods=1, closed='left').min()["RunTime"]
#         df[f'mean_time_{factor}'] = df_g["RunTime"].rolling(rolling_window, min_periods=1, closed='left').mean()["RunTime"]
#         df[f'min_split_time_{factor}'] = df_g["SplitTimes"].rolling(rolling_window, min_periods=1, closed='left').min()["SplitTimes"]
#         df[f'mean_split_time_{factor}'] = df_g["SplitTimes"].rolling(rolling_window, min_periods=1, closed='left').mean()["SplitTimes"]
#     return(df)