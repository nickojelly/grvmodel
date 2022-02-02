#pickle Load results
resultspicklefile = open( r'C:\Users\Nick\Documents\GitHub\grvmodel\Data Processed\results-df-w-dates.npy', 'rb')
resultsdf = pickle.load(resultspicklefile)
resultsdf = pd.DataFrame(resultsdf)
resultsdf.head()
split_distances = pd.read_csv("split_distance.csv")
resultsdf["dist"] = resultsdf["dist"].astype(str).str[:-1].astype(float)

resultsdf.run_time = resultsdf.run_time.astype(float)
resultsdf["track_id"] = resultsdf.apply(lambda s: track_id_gen(s["dist"],s["track"]), axis=1)
resultsdf_merged = pd.merge(resultsdf, split_distances, on="track_id")
form = resultsdf_merged
race_forms = form.groupby(["race_id"])
print(race_forms)
race_input_layer = []
race_classes = []

resultsdf_merged["date"] = pd.to_datetime(resultsdf_merged["date"], format="%d %b %y")
resultsdf_merged["speed"] = pd.to_numeric(resultsdf_merged["run_time"])/pd.to_numeric(resultsdf_merged["dist_x"])
resultsdf_merged["split_speed"] = pd.to_numeric(resultsdf_merged["split_times"])/pd.to_numeric(resultsdf_merged["split_dist_estim"])
resultsdf_merged["box"] = pd.to_numeric(resultsdf_merged["box"])
resultsdf_merged

for i,j in tqdm.tqdm(race_forms):
    try:
        race_date = pd.to_datetime(j["date"].iloc[0])
        #print(f"{race_date=}")
        dogs = j.dog_id
        #print(f"race id {i=}")
        race_dist = j.dist_x.iloc[1]
        dog_info = []
        i = 0
        for dog in dogs:
            #print(f"{dog=}")
            prev_form = form[(form.dog_id == dog)&(form["date"]< race_date)]
            dog_speed_avg = prev_form.speed.mean()
            dog_split_avg = prev_form.split_speed.mean()
            dog_race_count = prev_form.shape[0]
            dog_box = j.box.iloc[i]
            dog_stats = (dog, [dog_box,dog_speed_avg,dog_split_avg,dog_race_count], j.place.iloc[i])
            dog_info.append(dog_stats)
            #print(f"{dog_stats=}")
            i = i+1
        dog_info.sort(key=itemgetter(0))
        input_layer = [item for sublist in [x[1] for x in dog_info] for item in sublist]
        classes = [x[2] for x in dog_info]
        race_input_layer.append(input_layer)
        race_classes.append(classes)
    except Exception as e:
        print(f"{e=}")
