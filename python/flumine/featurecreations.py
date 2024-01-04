import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle
import tqdm
from operator import itemgetter
import math


def track_id_gen(dist, track):
    if track == None:
        return "NA"
    return track[0:4] + "-" + str(int(dist))


def generate_prediction_dataframe(prev_results_file, prediction_df):
    prev_results_file = "results-df-merged-prices.npy"

    form = pd.DataFrame(pickle.load(open(prev_results_file, "rb")))

    # prev_full_details = r"C:\Users\Nick\Documents\GitHub\grvmodel\Python\DATA\full_race_details_ft.csv"
    # prev_full_details = pd.read_csv(prev_full_details)
    ohe = OneHotEncoder(dtype=int, sparse=False, handle_unknown="ignore")
    transformed = ohe.fit_transform(form[['Track']]) 


    pred_track_OHE = ohe.transform(prediction_df[["Track"]])

    prediction_df["tracksOnehot"] = pred_track_OHE.tolist()
    prediction_df["dist"] = prediction_df["Distance"].astype(str).str[:-1].astype(float)

    race_pred_forms = prediction_df.groupby(["RaceId"], sort=False)

    form = form.sort_values(["dateF"])
    race_forms = form.groupby(["RaceId"])
    dog_forms = form.groupby(["DogId"])
    print(race_forms)
    race_input_layer = []
    race_classes = []
    full_details_new = []
    dog_stats_df = []
    n = 0
    for race, j in tqdm.tqdm(race_pred_forms):
        race_date = pd.to_datetime(j["Date"].iloc[0])
        race_time = j.RaceTime.iloc[0]
        dogs = j["DogId"]
        race_id = race
        race_grade = j.RaceGrade.iloc[0]
        race_dist = j.dist.iloc[0]
        trackOHE = j.tracksOnehot.iloc[0]
        track_name = j.Track.iloc[0]

        dog_info = []
        num_features = 16

        # Starts off with blank outfits of what we need to fill, are then filled using box pos as indexer
        blank_stats = [-1] * num_features
        blank_race = [blank_stats] * 8
        blank_classes = [8] * 8
        blank_margins = [80] * 8
        blank_sp = [0] * 8
        blank_bsp = [0] * 8
        blank_dogname = ["blank"] * 8

        i = 0
        for dog in dogs:
            if int(j.box.iloc[i]) > 8:
                #print(j.box.iloc[i])
                continue
            try:
                prev_form = dog_forms.get_group(dog)
                # print(f"{len(prev_form)=}")
                # No need to filter on date as form is today
                prev_form = prev_form[(prev_form["dateF"] < race_date)]
                # print(f"After date cut: {len(prev_form)=}")
                prev_races = len(prev_form)
                

                dog_speed_avg = prev_form.speed.mean()
                dog_speed_sd = prev_form.speed.std()
                dog_speed_max = prev_form.speed.max()
                dog_split_avg = prev_form.split_speed.mean()
                dog_split_sd = prev_form.split_speed.std()
                dog_split_max = prev_form.split_speed.max()
                dog_split_mg_avg = prev_form.split_margins.mean()
                dog_margin_mean = prev_form.margin.mean()
                dog_margin_sd = prev_form.margin.std()
                if prev_races > 0:
                    previous_race = prev_form.RaceId.iloc[0]
                    dog_weight = prev_form.weigth.iloc[-1]
                    dog_margin_last = prev_form.margin.iloc[0]
                    dog_speed_last = prev_form.speed.iloc[0]
                    dog_race_count = prev_form.shape[0]
                    try:
                        dog_wins = len(prev_form[prev_form["place"] == 1])
                        dog_places = len(prev_form[prev_form["place"] <= 3])
                    except Exception as e:
                        print("no win")
                else:
                    dog_weight = -1
                    dog_margin_last = -1
                    dog_speed_last = -1
                    dog_race_count = 0
                    dog_wins = 0
                    dog_places = 0
                    previous_race = -1
                # print(dog_margin_last)
                # print()

                dog_box = int(j.box.iloc[i])
                # dog_weight = j.weigth.iloc[i] # weight not in forms for prediction going to use last weight
                stats = [
                    dog_box,
                    dog_speed_avg,
                    dog_speed_sd,
                    dog_speed_max,
                    dog_split_avg,
                    dog_split_sd,
                    dog_split_max,
                    dog_split_mg_avg,
                    dog_margin_mean,
                    dog_margin_sd,
                    dog_race_count,
                    dog_wins,
                    dog_places,
                    dog_weight,
                    dog_speed_last,
                    dog_margin_last,
                ]
                # print(stats)
                stats = [-1 if math.isnan(x) else x for x in stats]

                # Fill in the blank outputs
                idx = dog_box - 1
                blank_race[idx] = stats
                blank_dogname[idx] = j.DogName.iloc[i]

                # dog_stats_df.append(
                #     [
                #         dog,
                #         j.DogName.iloc[i],
                #         race_id,
                #         race_grade,
                #         race_date,
                #         trackOHE,
                #         track_name,
                #         race_dist,
                #         stats,
                #         previous_race,
                #         race_time                        
                #     ]
                # )
                
            except Exception as e:
                stats = [int(j.box.iloc[i])] + [-1] * (num_features-1)
                previous_race = -1
                print(e)
            dog_stats_df.append(
                [
                    dog,
                    j.DogName.iloc[i],
                    race_id,
                    race_grade,
                    race_date,
                    trackOHE,
                    track_name,
                    race_dist,
                    stats,
                    previous_race,
                    j.RaceTime.iloc[0]
                ])

            i = i + 1
    dfx = pd.DataFrame(
        dog_stats_df,
        columns=[
            "dogid",
            "dog_name",
            "raceid",
            "race_grade",
            "date",
            "trackOHE",
            "track_name",
            "dist",
            "stats",
            "prev_race",
            "race_time"
        ],
    )

    return dfx

def generate_results_df(previous_results_file : str, new_FT_dog_data : pd.DataFrame ,new_FT_race_data : pd.DataFrame , betfair_sp_file :str, split_dist:str, mode="all", prev_df=None):
    previous_results = pd.read_pickle(previous_results_file)


    ohe = OneHotEncoder(dtype=int, sparse=False, handle_unknown="ignore")
    transformed = ohe.fit_transform(previous_results[['Track']]) 

    race_details = new_FT_race_data[['@id', 'RaceNum', 'Distance', 'date','RaceTime', 'RaceName', 'RaceGrade', 'Track']]
    race_details['RaceId'] = race_details['@id']
    
    dog_results = new_FT_dog_data.copy()
    dog_results['Place'] = pd.to_numeric(new_FT_dog_data['Place'], errors='coerce')
    print(dog_results.shape)
    dog_results = dog_results.dropna(subset=['Place'])
    print(dog_results.shape)

    dog_results['DogId'] = dog_results['@id']
    dog_results['box'] = dog_results['Box']

    dog_results['SplitTimes']= dog_results.SplitMargin.astype(float)
    dog_results['minMargin'] = dog_results.groupby('RaceId')['SplitTimes'].transform('min')
    dog_results['SplitMargin'] = dog_results.SplitTimes-dog_results.minMargin
    dog_results['Margin1'] = dog_results['Margin1'].fillna(0)
    dog_results['StartPrice'] = dog_results['StartPrice'].fillna(0)

    full_details = pd.merge(dog_results,race_details, how='left', on='RaceId')

    betfair_df = pickle.load(open(betfair_sp_file, 'rb'))
    betfair_df['dateF'] = pd.to_datetime(betfair_df.EVENT_DT, dayfirst=True).dt.date
    betfair_df['dog_name'] = betfair_df.dog.str[1:].str.upper()
    
    resultsdf = full_details
    # This was commented out as it was producing OHE of 61 instead of 74
    # ohe = OneHotEncoder(dtype=int, sparse=False, handle_unknown="ignore")
    # transformed = ohe.fit_transform(previous_results[['Track']]) 
    new_results_track_OHE = ohe.transform(resultsdf[["Track"]])
    resultsdf["tracksOnehot"] =  new_results_track_OHE.tolist()


    split_distances = pd.read_csv(split_dist)
    resultsdf["dist"] = resultsdf["Distance"].astype(str).str[:-1].astype(float)
    resultsdf = resultsdf[resultsdf['RunTime'].notnull()]
    resultsdf['run_time'] = pd.to_numeric(resultsdf['RunTime'])
    resultsdf['split_margins'] = resultsdf.SplitMargin.astype(float)

    resultsdf['place'] = resultsdf.Place.astype(float)
    resultsdf["track_id"] = resultsdf.apply(
        lambda s: track_id_gen(s["dist"], s["Track"]), axis=1
    )

    resultsdf_merged = pd.merge(resultsdf, split_distances, on=["track_id", "Track"], how='left')
    resultsdf_merged = resultsdf_merged[resultsdf_merged['RunTime'].notnull()]
    resultsdf_merged['split_dist_estim'] = resultsdf_merged['split_dist_estim'].fillna(100)
    resultsdf_merged["dateF1"] = pd.to_datetime(resultsdf_merged["date"], format="%d %b %y")
    resultsdf_merged["StartPrice"] = pd.to_numeric(resultsdf_merged['StartPrice'].str[1:-1])
    resultsdf_merged["speed"] = pd.to_numeric(resultsdf_merged["RunTime"]) / pd.to_numeric(
        resultsdf_merged["dist_x"]
    )

    resultsdf_merged["split_speed"] = pd.to_numeric(
        resultsdf_merged["SplitTimes"]
    ) / pd.to_numeric(resultsdf_merged["split_dist_estim"])

    resultsdf_merged["box"] = pd.to_numeric(resultsdf_merged["Box"])
    resultsdf_merged["margin"] = pd.to_numeric(resultsdf_merged["Margin1"])
    resultsdf_merged["weigth"] = pd.to_numeric(resultsdf_merged["Weight"])
    resultsdf_merged['split_margins'] = pd.to_numeric(resultsdf_merged.SplitMargin)
    resultsdf_merged["dateF"] = pd.to_datetime(resultsdf_merged["date"], format="%d %b %y").dt.date
    resultsdf_merged['dog_name'] = resultsdf_merged['DogName']

    resultsdf_merged = pd.merge(
        resultsdf_merged, betfair_df, how="left", on=["dateF", "dog_name"]
    )

    form = resultsdf_merged.sort_values("dateF", ascending=False)
    form.loc[form['place']==1, 'margin']=0

    all_results = pd.concat([previous_results, resultsdf_merged])

    all_results =   all_results.drop_duplicates( subset=['RaceId', 'DogId'])

    with open("results-df-merged-prices.npy", "wb") as fp:   #Pickling
    
        pickle.dump(all_results, fp)

    form = all_results.sort_values(['dateF'])
    dog_forms = form.groupby(["DogId"])

    if mode=="all":
        race_forms = all_results.groupby(["RaceId"])
    else:
        race_forms = resultsdf_merged.groupby(["RaceId"])
    
    print(race_forms)
    race_input_layer = []
    race_classes = []
    full_details_new = []
    dog_stats_df = []
    resultsdf_merged.head(50)
    n = 0

    for race, j in tqdm.tqdm(race_forms):
        try:
            race_date = pd.to_datetime(j["dateF"].iloc[0])
            dogs = j['DogId']
            race_id = race
            race_grade = j.RaceGrade.iloc[0]
            race_dist = j.dist_x.iloc[0]
            trackOHE = j.tracksOnehot.iloc[0]
            track_name = j.Track.iloc[0]
            dog_info = []
            num_features = 16
            

            #Starts off with blank outfits of what we need to fill, are then filled using box pos as indexer
            blank_stats = [-1]*num_features
            blank_race = [blank_stats]*8
            blank_classes = [8]*8
            blank_margins = [80]*8
            blank_sp = [0]*8
            blank_bsp = [0]*8
            blank_dogname = ["blank"]*8
            if len(dogs)>8:
                continue
            i = 0
            for dog in dogs:
                
                #print(f"dog id = {dog}")
                # if i ==1:
                #     print(f"{race_date=}")
                
                prev_form = dog_forms.get_group(dog)
                previous_race = prev_form.RaceId.iloc[0]
                #print(f"{len(prev_form)=}")
                prev_form = prev_form[(prev_form["dateF"] < race_date)]
                #print(f"After date cut: {len(prev_form)=}")
                prev_races = len(prev_form)


                dog_speed_avg = prev_form.speed.mean()
                dog_speed_sd = prev_form.speed.std()
                dog_speed_max = prev_form.speed.max()
                dog_split_avg = prev_form.split_speed.mean()
                dog_split_sd = prev_form.split_speed.std()
                dog_split_max = prev_form.split_speed.max()
                dog_split_mg_avg = prev_form.split_margins.mean()
                dog_margin_mean = prev_form.margin.mean()
                dog_margin_sd = prev_form.margin.std()
                if prev_races>0:
                    dog_margin_last = prev_form.margin.iloc[0]
                    dog_speed_last = prev_form.speed.iloc[0]
                    dog_race_count = prev_form.shape[0]
                    try:
                        dog_wins = len(prev_form[prev_form["place"] == 1])
                        dog_places = len(prev_form[prev_form["place"] <= 3])
                    except Exception as e:
                        print("no win")
                else:
                    dog_margin_last = -1
                    dog_speed_last = -1
                    dog_race_count = 0
                    dog_wins = 0
                    dog_places = 0
                    previous_race = -1
                # print(dog_margin_last)
                #print()

                dog_box = j.box.iloc[i]
                dog_weight = j.weigth.iloc[i] # weight not in forms for prediction going to use last weight
                stats = [
                        dog_box,
                        dog_speed_avg,
                        dog_speed_sd,
                        dog_speed_max,
                        dog_split_avg,
                        dog_split_sd,
                        dog_split_max,
                        dog_split_mg_avg,
                        dog_margin_mean,
                        dog_margin_sd,
                        dog_race_count,
                        dog_wins,
                        dog_places,
                        dog_weight,
                        dog_speed_last,
                        dog_margin_last
                        ]
                #print(stats)
                stats = [-1 if math.isnan(x) else x for x in stats]
                dog_stats = (
                    dog,
                    stats,
                    j.place.iloc[i],
                    j.StartPrice.iloc[i],
                    j.margin.iloc[i],
                    j.BSP.iloc[i],
                )

                #Fill in the blank outputs
                idx = dog_box-1
                blank_race[idx] = stats
                blank_classes[idx] = j.place.iloc[i]
                blank_bsp[idx] = j.BSP.iloc[i]
                blank_margins[idx] = j.margin.iloc[i]
                blank_sp[idx] = j.StartPrice.iloc[i]
                blank_dogname[idx] = j.dog_name.iloc[i]




                dog_info.append(dog_stats)
                dog_stats_df.append([dog,
                        j.dog_name.iloc[i],
                        race_id,
                        race_grade,
                        race_date,
                        trackOHE,
                        track_name,
                        race_dist,
                        stats,
                        j.place.iloc[i],
                        j.StartPrice.iloc[i],
                        j.margin.iloc[i],
                        j.BSP.iloc[i],
                        j.RunTime.iloc[i],
                        previous_race])
                #print(f"{dog_stats_df=}")
                i = i + 1
            
        except Exception as e:
            print(f"{e=}")

    dfx = pd.DataFrame(dog_stats_df, columns=['dogid','dog_name','raceid','race_grade','date','trackOHE','track_name','dist','stats','place','startprice','margin','bfSP', 'runtime', 'prev_race'])

    if prev_df:
        dfx = pd.concat(prev_df, dfx)
        dfx = dfx.drop_duplicates()

    return dfx