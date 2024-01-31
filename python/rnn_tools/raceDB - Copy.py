import pickle
import polars as pl
import pandas as pd
from rnn_tools.rnn_classes import *

def neg_identity(input):
    return -1*(input)

def boosted_softmin(input):
    sft_min = nn.Softmin(dim=-1)
    return sft_min(torch.exp(input))

print(os.getcwd())

import concurrent.futures

def chunks(xs, n):
    n = max(1, n)
    chunk_size = len(xs) // n
    return [xs[i : i + chunk_size] for i in range(0, len(xs), chunk_size)]

def process_data(data):
    dogs = {}
    for dog_id, dog_data in data:
        dog_data["next_race"] = dog_data["raceid"].shift(-1).fillna(-1)
        dog = Dog(dog_id, dog_data.dog_name, 64, 1) 
        dogs[dog_id] = dog
        dog_data.apply(lambda x: dogs[dog_id].add_races(x['raceid'], x['date'],x['stats_cuda'],x['next_race'], x['prev_race'], x['box'],x['margin'], x['bfSP'], x['StartPrice']), axis=1)
    return dogs

def process_race(data,null_dog_i, raceDB, margin_fn, device, v6=False):
    races = {}
    # print(list(raceDB.dogsDict.items()))
    for i,j in tqdm(data):
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
            # raceDB.racesDict[i].win_weight = track_weights[j.track_name.iloc[0]][dog_win_box-1]
            # raceDB.racesDict[i].weights = track_weights[j.track_name.iloc[0]]
            # raceDB.racesDict[i].margin_weights = margin_weights[(j.track_name.iloc[0],j.dist.iloc[0])]
            # raceDB.racesDict[i].win_margin_weight = margin_weights[(j.track_name.iloc[0],j.dist.iloc[0])][dog_win_box-1]
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



def process_all_data(df_g, raceDB):
    groups = list(df_g)
    chunked_groups = list(chunks(groups, 8))
    print(len(chunked_groups))
    [print(i,len(x)) for i,x in enumerate(chunked_groups)]
    _process_jobs = []
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for i, m in enumerate(chunked_groups):
            _process_jobs.append(
                executor.submit(process_data, m)
            )
        print(_process_jobs)
    for job in tqdm(concurrent.futures.as_completed(_process_jobs), total=len(_process_jobs)):
            results.append(job.result())  

    return results

def process_all_races(df_g, raceDB):
    groups = list(df_g)
    chunked_groups = list(chunks(groups, 8))
    print(len(chunked_groups))
    [print(i,len(x)) for i,x in enumerate(chunked_groups)]
    _process_jobs = []
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for i,m in enumerate(chunked_groups):
            _process_jobs.append(
                executor.submit(process_race, m, raceDB.null_dog_i, raceDB, raceDB.margin_fn, raceDB.device, raceDB.v6)
            )

def build_dataset(data, hidden_size, state_filter=None, margin_type='sftmin', test_date=None, v6=False, date_filter=None, track_filter=None, device='cuda:0')-> Races:

    # with open(data, 'rb') as f:
    #     data, stats_cols = pickle.load(f)

    # dog_stats_df = pd.DataFrame(data)

    dog_stats_df = pd.read_feather(data)
    stats_cols = dog_stats_df['stats_cols'].iloc[0]

    print(stats_cols)
    dog_stats_df = dog_stats_df.drop_duplicates(subset=['dogid', 'raceid'])
    print(dog_stats_df.shape)
    print(len(dog_stats_df.stats.iloc[0]))
    dog_stats_df['stats_cuda'] = dog_stats_df.apply(lambda x: torch.tensor(x['stats']), axis =1)
    # dog_stats_df['box'] = dog_stats_df['stats'].apply(lambda x: x[0])
    dog_stats_df['runtime'] = pd.to_numeric(dog_stats_df['runtime'])
    dog_stats_df.loc[dog_stats_df['place']==1, 'margin']=0

    print(f"Latest date = {pd.to_datetime(dog_stats_df.date).max()}")

    if state_filter:
        if isinstance(state_filter, list):
            dog_stats_df = dog_stats_df[dog_stats_df['state'].isin(state_filter)].reset_index()
            print(f'size after state filter {dog_stats_df.shape}')
            if track_filter:
                dog_stats_df = dog_stats_df[dog_stats_df['track_name'].str.contains(track_filter, na=False)].reset_index()
                print(f'size after track filter {dog_stats_df.shape}')
        else:
            dog_stats_df = dog_stats_df[dog_stats_df['state'].str.contains(state_filter, na=False)].reset_index()
        print(dog_stats_df.shape)
    dog_stats_df = dog_stats_df.reset_index(drop=True)

    if date_filter:
        dog_stats_df = dog_stats_df[dog_stats_df['date']>date_filter]

    print(f"Latest date = {pd.to_datetime(dog_stats_df.date).max()}")

    #Generate weights for classes per track:%APPDATA%\Code\User\settings.json

    grouped = dog_stats_df.groupby('track_name')
    track_weights = {}

    for i,j in grouped:
        weights = (1-(j[j['place']==1]['box'].value_counts(sort=False)/len(j[j['place']==1]))).tolist()
        if len(weights) !=8:
            weights.append(0)
        track_weights[i] = torch.tensor(weights).to(device)

    grouped = dog_stats_df.groupby(['track_name','dist'], as_index=False)

    grouped_track_box = dog_stats_df.groupby(['track_name', 'box', 'dist'], as_index=False)
    x = grouped_track_box.margin.sum()
    track_margin_sum = grouped.margin.sum().reset_index()
    x_r = x.reset_index().merge(track_margin_sum, how='left', on=['track_name','dist'])
    x_r['adj'] = x_r['margin_x']/x_r['margin_y']
    x_r_g = x_r.groupby(['track_name','dist'], as_index=False)
    margin_weights = {}
    for i,j in x_r_g:
        test = j
        #break
        track = i[0]
        dist = i[1]
        #margin_weights[track] = {}
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
    # dog_stats_df = pl.from_pandas(dog_stats_df)
    dog_stats_group = dog_stats_df.groupby("dogid", sort=False, as_index=False)
    # dog_stats_df["next_race"] = dog_stats_df['raceid'].shift(-1).fillna(-1)
    unique_dogs = dog_stats_df.drop_duplicates(subset='dogid')['dogid']
    raceDB.dog_ids = unique_dogs.tolist()
    # unique_dogs.apply(lambda x: raceDB.add_dog(x['dogid'], x['dog_name']), axis=1)
    #dog_stats_group.apply(lambda x: raceDB.dogsDict[i].add_races(x['raceid'], x['date'], torch.Tensor(x['stats']),x['next_race'], x['prev_race'], x['box'], x['bfSP'], 0), axis=1)
    result = process_all_data(dog_stats_group, raceDB)
    dogs_dict = {}
    [dogs_dict.update(x) for x in result]
    raceDB.dogsDict = dogs_dict
    # if "prev_race" in dog_stats_df.columns:
    #     for i,j in tqdm(dog_stats_group):
    #         j["next_race"] = j["raceid"].shift(-1).fillna(-1)
    #         # raceDB.add_dog(i, j.dog_name.iloc[0])
    #         # j.sort_values(['date'])
    #         j.apply(lambda x: raceDB.dogsDict[i].add_races(x['raceid'], x['date'],x['stats_cuda'],x['next_race'], x['prev_race'], x['box'],x['margin'], x['bfSP'], x['StartPrice']), axis=1)
    # else:
    #     raise Exception("Data issues encountered.")
    # #Fill in races portion
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
