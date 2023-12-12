# Import libraries for logging in
import betfairlightweight
from flumine import Flumine, clients
import logging 
import re
import pandas as pd
from datetime import datetime

if __name__=="__main__":


    
    trading = betfairlightweight.APIClient('nickbarlow@live.com.au','76ff98a6',app_key='JFWqJHqB4Akfi5hK')
    client = clients.BetfairClient(trading, interactive_login=True)
    trading.login_interactive()


    racing_filter=betfairlightweight.filters.market_filter(
        event_type_ids=["4339"], # Greyhounds
        market_countries=["AU"], # Australia
        market_type_codes=["WIN"], # Win Markets
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
        max_results=100,
    )

    df_list = []
    for i in results:
        print(f"{i.market_id,i.market_name,i.market_start_time.hour, i.market_start_time.minute, i.event.venue,  i.description.market_type} ")
        market_id = i.market_id
        race_num = int(re.sub("[^0-9]", "", i.market_name.split(' ',1)[0] ))
        track =  i.event.venue
        dist = i.market_name.split(' ',2)[0]
        print(race_num)
        for dog in i.runners:
            print(f"id = {dog.selection_id}, name = {dog.runner_name.split(' ',1)[1].upper()}")
            df_list.append([market_id, track, dist, race_num, dog.selection_id, dog.runner_name.split(' ',1)[1].upper()])
        print(df_list)
    df = pd.DataFrame(data = df_list, columns=['market_id', 'track', 'dist', 'race_num', 'runner_id', 'runnner_name'])

    today = datetime.today().strftime('%Y-%m-%d')

    df.to_pickle(f'betfair races {today}.npy')