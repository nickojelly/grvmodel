import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle
import tqdm
from operator import itemgetter
import math
import polars as pl
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

MUTUAL_BEST_20 = ['speed_max_30', 'split_speed_max_30', 'speed_max_5',
       'split_speed_max_5', 'speed_avg_1', 'speed_max_1',
       'split_speed_avg_1', 'split_speed_max_1', 'speed_avgdist_10',
       'speed_maxdist_10', 'split_speed_maxdist_10', 'speed_maxbox_10',
       'split_speed_maxbox_10', 'speed_maxtrack_box_10',
       'split_speed_maxtrack_box_10', 'speed_avgtrack_dist_10',
       'speed_maxtrack_dist_10', 'split_speed_maxtrack_dist_10',
       'min_time_track_dist', 'mean_time_track_dist']

REQ_COLUMNS = ['DogId',
               'DogName',
               'RaceId',
               'RaceGrade',
               'dateF',
               'date',
               'RaceTime',
               'RaceNum',
               'Track',
               'State',
               'track_code',
               'dist_x',
               'dist_round',
               'BSP',
               'Place',
               'place',
               'margin',
               'box',
               'speed',
               'split_speed',
               'split_margins',
               'MarginTime',
               'run_home_speed_v1',
               'run_home_speed',
               'split_speed_v1',
               'RunHomeTime',
               'first_out',
               'PIR',
               'PIR_adj',
               'start_pos',
               'pos_change',
               'weight',
               'RunTime',
               'SplitTimes',
               'StartPrice',
               'DogGrade']

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
    df[f'prev_race'] = df_g['RaceId'].shift(1).fillna('-1').astype('string')
    df[f'prev_race_date'] = df_g['date'].shift(1).fillna('-1').astype('string')
    df[f'prev_race_track'] = df_g['Track'].shift(1).fillna('-1').astype('string')
    df[f'prev_race_state'] = df_g['State'].shift(1).fillna('-1').astype('string')
    df[f'next_race'] = df_g['RaceId'].shift(-1).fillna('-1').astype('string')
    return(df)

def generate_prev_weight(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    df[f'prev_weight'] = df_g['weight'].shift(1).fillna(-1)
    return(df)

def generate_prediction_dataframe_v2(prev_results_file:str, prediction_df, ohe='encoder_new',simple=False):
    #prev_results_file = "results-df-merged-prices.npy"

    form = pd.read_feather(prev_results_file)

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



    
    form = form[form['DogId'].isin(prediction_df['DogId'])]

    form['win'] = form['place'].apply(lambda x: 1 if x ==1 else 0)
    form['count'] = 1
    form['box_location'] = form['box'].apply(lambda x: [1,0,0] if x<3 else([0,1,0] if x<7 else [0,0,1]))
    form['inside'] = form['box'].apply(lambda x: 1 if x<3 else 0)
    form['midfield'] = form['box'].apply(lambda x: 1 if 2<x<7 else 0)
    form['wide'] = form['box'].apply(lambda x: 1 if 6<x else 0)

    form = form.reset_index(drop=True)
    dog_form = form.groupby(['DogId'], sort=False, as_index=False)
    dist_form = form.groupby(['DogId','dist_round'], sort=False, as_index=False)
    box_form = form.groupby(['DogId','box'], sort=False, as_index=False)
    track_box_form = form.groupby(['DogId','box','Track'], sort=False, as_index=False)
    track_dist_form = form.groupby(['DogId','dist_x','Track'], sort=False, as_index=False)



    form = form.reset_index(drop=True)
    dog_form = form.groupby(['DogId'], sort=False, as_index=False)


    form = generate_prev_weight(form, dog_form)
    if simple:
        form_stats = generate_dog_stats_simple(form, dog_form,rolling_window=1)
        print("fin 1")
    else:
        dist_form = form.groupby(['DogId','dist_round'], sort=False, as_index=False)
        box_form = form.groupby(['DogId','box'], sort=False, as_index=False)
        track_box_form = form.groupby(['DogId','box','Track'], sort=False, as_index=False)
        track_dist_form = form.groupby(['DogId','dist_x','Track'], sort=False, as_index=False)
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

    form_only_stats = form_stats[['box','inside','midfield','wide','prev_weight','DogGrade']+stats_cols].fillna(-1.0).values
    stats = pd.Series(form_only_stats.tolist())

    form = form_stats.drop(columns=stats_cols)
    form['stats'] = stats

    form = generate_prev_race(form, dog_form)

    form = form[['DogId', 'DogName', 'RaceId','race_num','RaceGrade', 'date', 'race_time','tracksOnehot','Track','State','track_code','dist_x', 'BSP','RunTime','prev_race','prev_race_date','prev_race_track','Place','margin','box','stats']]
    form.columns=['dogid', 'dog_name', 'raceid','race_num','race_grade','date','race_time', 'trackOHE','track_name', 'state', 'track_code','dist','bfSP','runtime','prev_race', 'prev_race_date','prev_race_track', 'place', 'margin','box','stats']

    return form,stats_cols

def generate_results_df_v2(previous_results_file, new_FT : pd.DataFrame ,new_FT_race_data : pd.DataFrame, betfair_sp_file :str, split_dist:str, mode="all", prev_df=None, ohe='encoder_new', simple=False,normalize=False,v6=False):
    ohe = pickle.load(open(ohe, 'rb'))

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
    dog_results['RunTime'] = pd.to_numeric(dog_results['RunTime'])
    dog_results['MinTime'] = dog_results.groupby('RaceId')['RunTime'].transform('min')
    dog_results['MarginTime'] = pd.to_numeric(dog_results.RunTime) - dog_results.MinTime
    dog_results['Margin1'] = pd.to_numeric(dog_results['Margin1'], errors='coerce').fillna(0)
    dog_results['StartPrice'] = dog_results['StartPrice'].fillna(0)
    dog_results['DogGrade'] = pd.to_numeric(dog_results['DogGrade'], errors='coerce').fillna(8)

    # dog_results['RunHomeTime'] =  dog_results.SplitTimes-dog_results.minMargin

    dog_results['PIR_adj'] = dog_results.PIR.str.replace('[^0-9]', '',regex=True).fillna('0')
    dog_results['start_pos'] =  pd.to_numeric(dog_results['PIR_adj'].apply(lambda x: x[0] if x else 8))
    dog_results['last_pos'] =  pd.to_numeric(dog_results['PIR_adj'].apply(lambda x: x[-1] if x else 8))
    dog_results['first_out'] = dog_results['start_pos'].apply(lambda x: 1 if x==1 else 0)
    dog_results['pos_change'] = pd.to_numeric(dog_results['start_pos']) - pd.to_numeric(dog_results['last_pos'])

    # full_details = pd.merge(dog_results,race_details, how='left', on='RaceId')
    
    betfair_df = pd.read_feather(betfair_sp_file)
    betfair_df['dateF'] = pd.to_datetime(betfair_df.EVENT_DT, dayfirst=True).dt.date
    betfair_df['dog_name'] = betfair_df.dog.str[1:].str.upper()

    print(f"{betfair_df['dateF'].max()=}")
    # asdfasdd
    resultsdf = dog_results
    print(len(resultsdf))

    # new_results_track_OHE = ohe.transform(resultsdf[["Track"]])
    # resultsdf["tracksOnehot"] =  new_results_track_OHE.tolist()
    print(resultsdf.columns)


    split_distances = pd.read_csv(split_dist)
    resultsdf["dist"] = resultsdf["Distance"].astype(str).str[:-1].astype(float)
    resultsdf = resultsdf[resultsdf['RunTime'].notnull()]
    resultsdf['RunTime'] = pd.to_numeric(resultsdf['RunTime'])
    resultsdf['RunHomeTime'] =  resultsdf.RunTime - resultsdf.SplitTimes
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
    resultsdf_merged["split_speed_v1"] = pd.to_numeric(resultsdf_merged["split_dist_estim"])/pd.to_numeric(resultsdf_merged["SplitMargin"])

    resultsdf_merged['run_home_speed'] = pd.to_numeric((resultsdf_merged["dist_x"]-resultsdf_merged["split_dist_estim"])/pd.to_numeric(resultsdf_merged.RunTime-resultsdf_merged.SplitTimes))
    resultsdf_merged['run_home_speed_v1'] = pd.to_numeric((resultsdf_merged["dist_x"]-resultsdf_merged["split_dist_estim"])/pd.to_numeric(resultsdf_merged.RunTime-resultsdf_merged.SplitMargin))

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


    previous_results = pd.read_feather(previous_results_file)
    previous_results.reset_index(drop=True,inplace=True)

    resultsdf_merged = resultsdf_merged[REQ_COLUMNS].reset_index(drop=True)

    all_results = pd.concat([previous_results, resultsdf_merged])
    all_results["StartPrice"] = pd.to_numeric(all_results['StartPrice'].apply(lambda x: x.replace('$','').replace('F','') if isinstance(x,str) else x), errors='coerce').fillna(0)
    all_results["StartProb"] = 1/all_results["StartPrice"]
    all_results['RunTime'] = pd.to_numeric(all_results['RunTime'])
    # all_results = pd.concat([previous_results, resultsdf_merged])

    all_results['race_time'] = pd.to_datetime(all_results['RaceTime'])

    all_results =   all_results.drop_duplicates( subset=['RaceId', 'DogId'], keep='last')


    all_results["tracksOnehot"] = [x for x in ohe.transform(all_results[["Track"]])]
    form = all_results.sort_values("dateF", ascending=True)
    form.loc[form['place']==1, 'margin']=0    

    form['win'] = form['place'].apply(lambda x: 1 if x ==1 else 0)
    form['count'] = 1
    form['box_location'] = form['box'].apply(lambda x: [1,0,0] if x<3 else([0,1,0] if x<7 else [0,0,1]))
    form['inside'] = form['box'].apply(lambda x: 1 if x<3 else 0)
    form['midfield'] = form['box'].apply(lambda x: 1 if 2<x<7 else 0)
    form['wide'] = form['box'].apply(lambda x: 1 if 6<x else 0)

    #For OG only

    form['first_out'] = form['start_pos'].apply(lambda x: 1 if x==1 else 0)

    all_results['RaceId'] = all_results['RaceId'].astype('category')
    all_results['DogId'] = all_results['DogId'].astype('category')
    all_results['RaceId'] = all_results['RaceId'].astype('category')

    all_results.reset_index(drop=True).to_feather(previous_results_file)
    # with open("results-df-merged-prices_cut.npy", "wb") as fp:   #Pickling
    
    #     pickle.dump(all_results, fp)


    print("starting stats rolling window")


    #NEW Groups for functions
    form = form.reset_index(drop=True)
    print(f"{form.shape=}")
    # form = form[form['State']=="VIC"].reset_index()
    print(f"{form.shape=}")
    form['box_g'] = form['box']
    form['dist_g'] = form['dist_x']
    dog_form = form.groupby(['DogId'], sort=False, as_index=False)

    form = pl.from_pandas(form)
    dog_form = form.groupby(['DogId'])
    

    print(f"{form.columns=}")

    if simple:
            form_stats = generate_dog_stats_simple(form, dog_form,rolling_window=1)
            print("fin 1")
    else:
        print(form.columns)
        dist_form = form.groupby(['DogId','dist_round'])
        box_form = form.groupby(['DogId','box_g'])
        track_box_form = form.groupby(['DogId','box_g','Track'])
        track_dist_form = form.groupby(['DogId','dist_g','Track'])
        form = generate_prev_weight(form, dog_form)
        # form_stats = generate_dog_stats(form, dog_form,rolling_window=30)
        # print("fin 30")
        # form_stats = generate_dog_stats(form_stats, dog_form,rolling_window=5)
        # print("fin 5")
        form_stats = generate_dog_stats_simple(form, dog_form,rolling_window=1)
        print("fin 1")
        form_stats = generate_dog_stats_simple(form_stats, dist_form,rolling_window=10,factor='dist')
        print("fin 10_dist")
        form_stats = generate_dog_stats_simple(form_stats, box_form,rolling_window=10,factor='box')
        print("fin 10 box")
        form_stats = generate_dog_stats_simple(form_stats, track_box_form,rolling_window=10,factor='track_box')
        print("fin 10 track_box")
        form_stats = generate_dog_stats_simple(form_stats, track_dist_form,rolling_window=10,factor='track_dist')
        print("fin 10 track_dist")  

    #Status columns hold all col names of generated stats
    stats_cols = ['inside','midfield','wide','weight','DogGrade']+[x for x in form_stats.columns if x not in form.columns]
    print(f"STATS COLS {stats_cols}")

    #Keep mutual best 20
    # stats_cols = stats_cols[[MUTUAL_BEST_20]]

    form_only_stats_w_track = form_stats[['Track','dateF','dist_x']+stats_cols].fillna(-1.0)

    form_only_stats_w_track.groupby(['Track', 'dist_x'], as_index=False).mean(numeric_only=True).reset_index().to_feather('expected stat values.fth')
    form_only_stats_w_track.reset_index().to_feather('all_stat_values.fth')
    # form_only_stats = form_stats[stats_cols].fillna(-1.0)
    form_only_stats = form_stats[['box']+stats_cols].fillna(-1.0)
    # form_only_stats = form_stats[['box','weight','DogGrade']+stats_cols].fillna(-1.0)
    stats = pd.Series(form_only_stats.values.tolist())
    
    if normalize:
        print("normalizing")
        form_only_stats_normed = (form_only_stats-form_only_stats.mean())/form_only_stats.std()
        form_only_stats_normed[['inside','midfield','wide','first_out_avg_1']] = form_stats[['inside','midfield','wide','first_out_avg_1']].fillna(-1.0)
        stats = pd.Series(form_only_stats_normed.values.tolist())
        form['stats'] = stats
        form['dist_x'] = (form['dist_x']-form['dist_x'].mean())/form['dist_x'].std()
        form_only_stats_normed.to_feather('all_stat_values_normed.fth')
        # form_stats.to_csv('form data for R analasis.csv')
    else:
        form = form_stats.drop(columns=stats_cols)
        form['stats'] = stats

    form = generate_prev_race(form, dog_form)

    form = form[['DogId', 'DogName', 'RaceId', 'RaceGrade', 'dateF','race_time','RaceNum','tracksOnehot','Track','State','track_code','dist_x', 'BSP',"StartPrice",'RunTime','prev_race','prev_race_date','prev_race_track','prev_race_state','Place','margin','box','stats']]
    form.columns=['dogid', 'dog_name', 'raceid', 'race_grade', 'date','race_time','race_num','trackOHE','track_name', 'state', 'track_code','dist','bfSP',"StartPrice",'runtime','prev_race', 'prev_race_date','prev_race_track','prev_race_state', 'place', 'margin','box','stats']

    return (form,stats_cols)

import polars as pl

def generate_dog_stats_simple(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    try:
        df = df.with_columns(pl.col('dist_x').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_dist_last__{rolling_window}'))
    except:
        df = df.with_columns(pl.lit(0).alias(f'{factor}_dist_last__{rolling_window}'))
    try:
        df = df.with_columns(pl.col('box').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_box_last__{rolling_window}'))
    except:
        df = df.with_columns(pl.lit(0).alias(f'{factor}_box_last__{rolling_window}'))

    df = df.with_column('box', df['box'].astype('int'))
    new_cols = [pl.col('speed').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_speed_avg_{rolling_window}'),
    # .alias(f'{factor}_split_speed_v1_{rolling_window}')) _ pl.col('split_speed_v1').rolling_mean(window_size=rolling_window, min_periods=1, closed='left')['split_speed_v1',
    pl.col('split_speed').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_split_speed_avg_{rolling_window}'),
    pl.col('split_margins').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_split_margin_avg_{rolling_window}'),
    pl.col('margin').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_margin_avg_{rolling_window}'),
    pl.col('MarginTime').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_margin_time_avg_{rolling_window}'),
    pl.col('RunHomeTime').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_RunHomeTime_{rolling_window}'),
    pl.col('run_home_speed').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_run_home_speed_{rolling_window}'),
    pl.col('first_out').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_first_out_avg_{rolling_window}'),
    pl.col('start_pos').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_pos_out_avg_{rolling_window}'),
    pl.col('pos_change').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_post_change_avg_{rolling_window}'),
    pl.col('win').cum_count().alias(f'{factor}_races_{rolling_window}'),
    pl.col('win').rolling_mean(window_size=1000, min_periods=1, closed='left').alias(f'{factor}_wins_{rolling_window}'),
    pl.col('win').rolling_sum(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_wins_last_{rolling_window}'),
    pl.col('weight').rolling_mean(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_weight_'),
    pl.col("RunTime").rolling_min(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_min_time_'),
    pl.col("SplitTimes").rolling_min(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_min_split_time_'),
    pl.col("StartPrice").rolling_min(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_last_start_price'),
    pl.col("StartProb").rolling_min(window_size=rolling_window, min_periods=1, closed='left').alias(f'{factor}_last_start_prob')]

    df = df.with_columns(new_cols)
    
    return(df)


