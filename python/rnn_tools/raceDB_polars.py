import pickle
import polars as pl
import pandas as pd
from rnn_tools.rnn_classes import *

def neg_identity(input):
    return -1*(input)

def boosted_softmin(input):
    sft_min = nn.Softmin(dim=-1)
    return sft_min(torch.exp(input))

def build_dataset(data, hidden_size, state_filter=None, margin_type='sftmin', test_date=None, v6=False, date_filter=None, track_filter=None, device='cuda:0')-> Races:

    dog_stats_df = pl.read_parquet(data)
    stats_cols = dog_stats_df['stats_cols'][0]
    # print(stats)

    print(stats_cols)
    dog_stats_df = dog_stats_df.unique(['dogid', 'raceid'])
    print(dog_stats_df.shape)
    print(len(dog_stats_df['stats'][0]))
    # dog_stats_df = dog_stats_df.with_columns(pl.col('stats').apply(lambda x: torch.tensor(x), return_dtype=pl.Object))
    dog_stats_df = dog_stats_df.with_columns(pl.col('runtime').cast(pl.Float32))
    dog_stats_df = dog_stats_df.with_columns(pl.when(pl.col('place') == 1).then(0).otherwise(pl.col('margin')).alias('margin'))

    print(f"Latest date = {dog_stats_df['date'].cast(pl.Date).max()}")

    if state_filter:
        if isinstance(state_filter, list):
            dog_stats_df = dog_stats_df.filter(pl.col('state').is_in(state_filter))
            print(f'size after state filter {dog_stats_df.shape}')
            if track_filter:
                dog_stats_df = dog_stats_df.filter(pl.col('track_name').str_contains(track_filter))
                print(f'size after track filter {dog_stats_df.shape}')
        else:
            dog_stats_df = dog_stats_df.filter(pl.col('state').str_contains(state_filter))
        print(dog_stats_df.shape)

    if date_filter:
        dog_stats_df = dog_stats_df.filter(pl.col('date') > date_filter)

    print(f"Latest date = {dog_stats_df['date'].cast(pl.Date).max()}")

    grouped = dog_stats_df.groupby('track_name').agg(
        pl.col('place').filter(pl.col('place')==1).alias('place'),
        pl.col('box').alias('box')
    )

    track_names = dog_stats_df['track_name'].unique().to_list()

    def calculate_weights(df):
        weights = (1 - df['box'].value_counts(sort=False) / len(df['place'])).tolist()
        if len(weights) != 8:
            weights.append(0)
        return weights

    track_weights = {track_name: torch.tensor(calculate_weights(dog_stats_df[dog_stats_df['track_name'] == track_name])).to(device) for track_name in track_names}

    grouped = dog_stats_df.groupby(['track_name','dist']).apply(lambda df: df)

    grouped_track_box = dog_stats_df.groupby(['track_name', 'box', 'dist']).apply(lambda df: df)
    x = grouped_track_box['margin'].sum()
    track_margin_sum = grouped['margin'].sum().reset_index()
    x_r = x.reset_index().merge(track_margin_sum, how='left', on=['track_name','dist'])
    x_r['adj'] = x_r['margin_x']/x_r['margin_y']
    x_r_g = x_r.groupby(['track_name','dist']).apply(lambda df: df)
    margin_weights = {}
    for i,j in x_r_g:
        test = j
        track = i[0]
        dist = i[1]
        weights = j['adj'].tolist()
        if len(weights)!= 8:
            weights.append(0)
        margin_weights[i]  = torch.tensor(weights).to(device)

    #Created RaceDB
    raceDB = Races(hidden_size, 1)
    raceDB.stats_cols = stats_cols
    raceDB.states = state_filter

    raceDB.latest_date = pd.to_datetime(dog_stats_df.date).max()

    num_features_per_dog = len(dog_stats_df['stats'].iloc[0])
    print(f"{num_features_per_dog=}")

    #Fill in dog portion:
    dog_stats_df = dog_stats_df.sort_values(['date'])
    dog_stats_df = pl.from_pandas(dog_stats_df)
    dog_stats_group = dog_stats_df.groupby("dogid", sort=False, as_index=False)
    # dog_stats_df["next_race"] = dog_stats_df['raceid'].shift(-1).fillna(-1)
    unique_dogs = dog_stats_df.drop_duplicates(subset='dogid')['dogid']
    raceDB.dog_ids = unique_dogs.tolist()
    # unique_dogs.apply(lambda x: raceDB.add_dog(x['dogid'], x['dog_name']), axis=1)
    #dog_stats_group.apply(lambda x: raceDB.dogsDict[i].add_races(x['raceid'], x['date'], torch.Tensor(x['stats']),x['next_race'], x['prev_race'], x['box'], x['bfSP'], 0), axis=1)

    if "prev_race" in dog_stats_df.columns:
        for i,j in tqdm(dog_stats_group):
            j["next_race"] = j["raceid"].shift(-1).fillna(-1)
            raceDB.add_dog(i, j.dog_name.iloc[0])
            j.sort_values(['date'])
            j.apply(lambda x: raceDB.dogsDict[i].add_races(x['raceid'], x['date'],x['stats_cuda'],x['next_race'], x['prev_race'], x['box'],x['margin'], x['bfSP'], x['StartPrice']), axis=1)
    else:
        raise Exception("Data issues encountered.")
    #Fill in races portion
    if margin_type=='sftmin':
        margin_fn = nn.Softmin(dim=-1)
    elif margin_type=='boosted_sftmin':
        margin_fn = boosted_softmin
    elif margin_type=='raw':
        margin_fn = nn.Identity()
    elif margin_type=='neg_raw':
        margin_fn = neg_identity
    #lsoftmax = F.log_softmax(dim=1)
    races_group = dog_stats_df.groupby(['raceid'])

    null_dog = Dog("nullDog", "no_name", raceDB.hidden_size, raceDB.layers)
    raceDB.add_dog("nullDog", "no_name")
    null_dog = raceDB.dogsDict['nullDog']
    null_dog_i = DogInput("nullDog", "-1", torch.ones(num_features_per_dog)*-100, null_dog,0, torch.zeros(raceDB.hidden_size),100,0,0,hidden_size=hidden_size)
    null_dog.input = null_dog_i
    null_dog_i.nextrace(-1)
    null_dog_i.prevrace(-1)

    # return raceDB

    #TO FIX LATER PROPER BOX PLACEMENT #FIXED
    dog_stats_df = dog_stats_df.sort_values('date')
    races_group = dog_stats_df.groupby('raceid', sort=False)
    # print(list(raceDB.dogsDict.items()))
    for i,j in tqdm(races_group):
    #Track info tensors
        # i = i[0]
        dist = torch.tensor([j.dist.iloc[0]]) 
        trackOHE = torch.tensor(j.trackOHE.iloc[0])
        #margins
        empty_dog_list = [null_dog_i]*8
        empty_margin_list = [100]*8
        empty_log_margin_list = [3]*8
        empty_place_list = [8]*8
        empty_finish_list = [40]*8
        empty_price_list = [0]*8
        empty_start_price_list = [1000]*8
        untouched_margin = [20]*8

        places_list = [x for x in j["place"]]
        boxes_list = [int(x) for x in j['box']]
        margin_list = [x for x in j["margin"]]
        time_list = [x for x in j["runtime"]]
        price_list = [x for x in j['bfSP'].astype(float)]
        start_price_list = [x for x in j['StartPrice'].astype(float)]

        # empty_log_margin_list = np.log(max(margin_list)+1)

        # print(f"{x=}\n{i=},\n{j['dogid']=}")
        
        dog_list = [raceDB.dogsDict[x].races[i] for x in j["dogid"]]

        #adjustedMargin = [margin_list[x-1] for x in boxes_list]
        for n,x in enumerate(boxes_list):
            empty_margin_list[x-1] = margin_list[n]
            empty_log_margin_list[x-1] = margin_list[n]+1
            empty_dog_list[x-1] = dog_list[n]
            empty_place_list[x-1] = places_list[n]
            empty_finish_list[x-1] = time_list[n]
            empty_start_price_list[x-1] = start_price_list[n]
            empty_price_list[x-1] = price_list[n]
            untouched_margin[x-1] = margin_list[n]
        adjustedMargin = (margin_fn(torch.tensor(empty_margin_list))).to(device) # chage here


        raceDB.add_race(i,trackOHE,dist, adjustedMargin)
        try:
            dog_win_box = int(j[j['place']==1]['box'].iloc[0])
        except Exception as e:
            dog_win_box = 1
            # print('thorwing')
        
        raceDB.racesDict[i].add_dogs(empty_dog_list)
        if not v6:
            raceDB.racesDict[i].nn_input()
        raceDB.racesDict[i].one_hot_class = torch.zeros_like(adjustedMargin).scatter_(0, torch.tensor(dog_win_box-1).to(device),1)
        raceDB.racesDict[i].track_name = j.track_name.iloc[0]
        raceDB.racesDict[i].grade = j.race_grade.iloc[0]
        try:
            raceDB.racesDict[i].win_weight = track_weights[j.track_name.iloc[0]][dog_win_box-1]
            raceDB.racesDict[i].weights = track_weights[j.track_name.iloc[0]]
            raceDB.racesDict[i].margin_weights = margin_weights[(j.track_name.iloc[0],j.dist.iloc[0])]
            raceDB.racesDict[i].win_margin_weight = margin_weights[(j.track_name.iloc[0],j.dist.iloc[0])][dog_win_box-1]
            raceDB.racesDict[i].win_price_weight = torch.tensor(empty_price_list[dog_win_box-1]).to(device)
            raceDB.racesDict[i].win_price_weightv2 = (1-1/(max(torch.tensor(empty_price_list[dog_win_box-1]),torch.tensor(1)))).to(device)
        except Exception as e:
            print(e)
        raceDB.racesDict[i].raw_margins = empty_margin_list
        raceDB.racesDict[i].raw_places = empty_place_list
        raceDB.racesDict[i].untouched_margin = untouched_margin
        raceDB.racesDict[i].prices = empty_price_list
        raceDB.racesDict[i].start_prices = empty_start_price_list
        raceDB.racesDict[i].race_time = j.race_time.iloc[0]
        raceDB.racesDict[i].race_date = j.date.iloc[0]
        raceDB.racesDict[i].race_num = j.race_num.iloc[0]
    

    raceDB.race_prices_to_prob()
    raceDB.create_new_weights()

    print(f"number of races = {len(raceDB.racesDict)}, number of unique dogs = {len(raceDB.dogsDict)}")
    raceDB.create_test_split()
    return raceDB

