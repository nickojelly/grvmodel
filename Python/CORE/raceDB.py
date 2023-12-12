from rnn_classes import *
import pickle
import pandas as pd

def build_pred_dataset(data, hidden_size)-> Races:

    #Load in pickeled dataframe
    # resultsdf = pickle.load(data)
    dog_stats_df = data
    dog_stats_df = dog_stats_df.fillna(-1).drop_duplicates(subset=['dogid', 'raceid'])
    #dog_stats_df['stats_cuda'] = dog_stats_df.apply(lambda x: torch.tensor(x['stats']), axis =1)
    dog_stats_df['box'] = dog_stats_df['stats'].apply(lambda x: x[0])

    #Created RaceDB
    raceDB = Races(hidden_size, 1)

    #Fill in dog portion:

    dog_stats_group = dog_stats_df.sort_values(['date']).groupby(["dogid"])

    for i,j in tqdm(dog_stats_group):
        raceDB.add_dog(i, j.dog_name.iloc[0])
        j.apply(lambda x: raceDB.dogsDict[i].add_races(x['raceid'], x['date'], torch.Tensor(x['stats']),-1, x['prev_race'], x['box']), axis=1)

    #Fill in races portion
    softmin = nn.Softmin(dim=0)
    races_group = dog_stats_df.groupby(['raceid'])

    null_dog = Dog("nullDog", "no_name", raceDB.hidden_size, raceDB.layers)
    null_dog_i = DogInput("nullDog", "-1", torch.zeros(16), null_dog,0, torch.zeros(raceDB.hidden_size),0,0)
    null_dog_i.nextrace(-1)
    null_dog_i.prevrace(-1)

    null_dog_list = [null_dog] * 8
    #TO FIX LATER PROPER BOX PLACEMENT #FIXED

    races_group = dog_stats_df.groupby(['raceid'])
    for i,j in tqdm(races_group):
    #Track info tensors
        dist = torch.tensor([j.dist.iloc[0]]) 
        trackOHE = torch.tensor(j.trackOHE.iloc[0])

        empty_dog_list = [null_dog_i]*8
        boxes_list = [x for x in j['box']]      
        dog_list = [raceDB.dogsDict[x].races[i] for x in j["dogid"]]

        for n,x in enumerate(boxes_list):
            empty_dog_list[x-1] = dog_list[n]
        
        raceDB.add_race(i,trackOHE,dist)
        
        # List of Dog Input??
        raceDB.racesDict[i].add_dogs(empty_dog_list)
        raceDB.racesDict[i].nn_input()
        raceDB.racesDict[i].add_track_name(j.track_name.iloc[0])
        raceDB.racesDict[i].track_name = j.track_name.iloc[0]
        raceDB.racesDict[i].grade = j.race_grade.iloc[0]
        raceDB.racesDict[i].dist = j.dist.iloc[0]
        raceDB.racesDict[i].race_time = j.race_time.iloc[0]
        raceDB.racesDict[i].race_date = j.date.iloc[0]

    print(f"number of races = {len(raceDB.racesDict)}, number of unique dogs = {len(raceDB.dogsDict)}")
    return raceDB

def build_dataset(data, hidden_size, track_filer = "NZ")-> Races:

    #Load in pickeled dataframe
    #resultsdf = pickle.load(data)
    dog_stats_df = pd.read_pickle(data)
    print(dog_stats_df.shape)
    dog_stats_df = dog_stats_df.fillna(-1).drop_duplicates(subset=['dogid', 'raceid'])
    print(dog_stats_df.shape)
    dog_stats_df['stats_cuda'] = dog_stats_df.apply(lambda x: torch.tensor(x['stats']), axis =1)
    dog_stats_df['box'] = dog_stats_df['stats'].apply(lambda x: x[0])
    dog_stats_df['runtime'] = pd.to_numeric(dog_stats_df['runtime'])
    dog_stats_df.loc[dog_stats_df['place']==1, 'margin']=0


    #ridiculous but only thing that drops tracks with null name
    #dog_stats_df = dog_stats_df.drop(dog_stats_df[dog_stats_df['track_name'].isna()].index)

    dog_stats_df = dog_stats_df[dog_stats_df['track_name'].str.contains(track_filer, na=False)].reset_index()

    #Generate weights for classes per track:

    grouped = dog_stats_df.groupby('track_name')
    track_weights = {}

    for i,j in grouped:
        weights = (1-(j[j['place']==1]['box'].value_counts(sort=False)/len(j[j['place']==1]))).tolist()
        if len(weights) !=8:
            weights.append(0)
        track_weights[i] = torch.tensor(weights).to('cuda:0')

    #track_weights['Cranbourne'] = track_weights['Cranbourne']*5
    

    #Created RaceDB
    raceDB = Races(hidden_size, 1)

    num_features_per_dog = len(dog_stats_df['stats'][0])
    print(f"{num_features_per_dog=}")

    #Fill in dog portion:

    dog_stats_group = dog_stats_df.sort_values(['date']).groupby(["dogid"])
    if "prev_race" in j.columns:
        for i,j in tqdm(dog_stats_group):
            j["next_race"] = j["raceid"].shift(-1).fillna(-1)
            
            raceDB.add_dog(i, j.dog_name.iloc[0])
            j.apply(lambda x: raceDB.dogsDict[i].add_races(x['raceid'], x['date'], torch.Tensor(x['stats']),x['next_race'], x['prev_race'], x['box'], x['bfSP'], x['startprice']), axis=1)
    else:
        for i,j in tqdm(dog_stats_group):
            j["next_race"] = j["raceid"].shift(-1).fillna(-1)
            j["prev_race"] = j["raceid"].shift(1).fillna(-1)
            raceDB.add_dog(i, j.dog_name.iloc[0])
            j.apply(lambda x: raceDB.dogsDict[i].add_races(x['raceid'], x['date'], torch.Tensor(x['stats']),x['next_race'], x['prev_race'], x['box'], x['bfSP'], x['startprice']), axis=1)

    #Fill in races portion
    softmin = nn.Softmin(dim=-1)
    #lsoftmax = F.log_softmax(dim=1)
    races_group = dog_stats_df.groupby(['raceid'])

    null_dog = Dog("nullDog", "no_name", raceDB.hidden_size, raceDB.layers)
    null_dog_i = DogInput("nullDog", "-1", torch.zeros(num_features_per_dog), null_dog,0, torch.zeros(raceDB.hidden_size),0,0)
    null_dog_i.nextrace(-1)
    null_dog_i.prevrace(-1)

    #TO FIX LATER PROPER BOX PLACEMENT #FIXED
    dog_stats_df = dog_stats_df.sort_values('date')
    races_group = dog_stats_df.groupby(['raceid'])
    for i,j in tqdm(races_group):
    #Track info tensors
        dist = torch.tensor([j.dist.iloc[0]]) 
        trackOHE = torch.tensor(j.trackOHE.iloc[0])
        #margins
        empty_dog_list = [null_dog_i]*8
        empty_margin_list = [20]*8
        empty_log_margin_list = [3]*8
        empty_place_list = [8]*8
        empty_finish_list = [40]*8
        empty_price_list = [0]*8
        untouched_margin = [20]*8

        places_list = [x for x in j["place"]]
        boxes_list = [x for x in j['box']]
        margin_list = [x for x in j["margin"]]
        time_list = [x for x in j["runtime"]]
        price_list = [x for x in j['bfSP'].astype(float)]

        # empty_log_margin_list = np.log(max(margin_list)+1)
        
        dog_list = [raceDB.dogsDict[x].races[i] for x in j["dogid"]]

        #adjustedMargin = [margin_list[x-1] for x in boxes_list]
        for n,x in enumerate(boxes_list):
            empty_margin_list[x-1] = margin_list[n]
            empty_log_margin_list[x-1] = margin_list[n]+1
            empty_dog_list[x-1] = dog_list[n]
            empty_place_list[x-1] = places_list[n]
            empty_finish_list[x-1] = time_list[n]
            empty_price_list[x-1] = price_list[n]
            untouched_margin[x-1] = margin_list[n]
        adjustedMargin = softmin(torch.tensor(empty_margin_list)).to('cuda:0')

        raceDB.add_race(i,trackOHE,dist, adjustedMargin)
        try:
            dog_win_box = j[j['place']==1]['box'][0].item()
        except Exception as e:
            dog_win_box = 1
        raceDB.racesDict[i].win_weight = track_weights[j.track_name.iloc[0]][dog_win_box-1]
        # List of Dog Input??
        raceDB.racesDict[i].add_dogs(empty_dog_list)
        raceDB.racesDict[i].nn_input()
        raceDB.racesDict[i].track_name = j.track_name.iloc[0]
        raceDB.racesDict[i].grade = j.race_grade.iloc[0]
        raceDB.racesDict[i].weights = track_weights[j.track_name.iloc[0]]
        raceDB.racesDict[i].raw_margins = empty_margin_list
        raceDB.racesDict[i].raw_places = empty_place_list
        raceDB.racesDict[i].untouched_margin = untouched_margin
        raceDB.racesDict[i].prices = empty_price_list
        raceDB.racesDict[i].race_date = j.date.iloc[0]

    raceDB.race_prices_to_prob()

    print(f"number of races = {len(raceDB.racesDict)}, number of unique dogs = {len(raceDB.dogsDict)}")
    raceDB.create_test_split()
    return raceDB
