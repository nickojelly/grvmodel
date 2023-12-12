from flumine import Flumine, clients
from flumine import BaseStrategy 
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder
from flumine.markets.market import Market
import betfairlightweight
from betfairlightweight.filters import streaming_market_filter
from betfairlightweight.resources import MarketBook
import re
import pandas as pd
import numpy as np
import datetime
import logging
from nltk.tokenize import regexp_tokenize
from joblib import load
import os
import csv
import logging
from flumine.controls.loggingcontrols import LoggingControl
from flumine.order.ordertype import OrderTypes
# import logging
#
from flumine.worker import BackgroundWorker
from flumine.events.events import TerminationEvent

class FlatBetting(BaseStrategy):
    def start(self) -> None:
        print("starting strategy 'FlatBetting' using the model we created the Greyhound modelling in Python Tutorial")

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        #print(market_book.status)
        if market_book.status != "CLOSED":
            return True

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        # Convert dataframe to a global variable
        global todays_data
        global tracks
        track = market.market_catalogue.event.venue

        # At the 60 second mark:
        if market.seconds_to_start < 30 and market_book.inplay == False and track in tracks:
            # get the list of dog_names, name of the track/venue and race_number/RaceNum from Betfair Polling API
            dog_names = []
            track = market.market_catalogue.event.venue
            race_number = market.market_catalogue.market_name.split(' ',1)[0]  # comes out as R1/R2/R3 .. etc
            race_number = re.sub("[^0-9]", "", race_number)  # only keep the numbers 
            race_number = int(race_number)
            for runner_cata in market.market_catalogue.runners:
                dog_name = runner_cata.runner_name.split(' ',1)[1].upper()
                dog_names.append(dog_name)

            # Use both the polling api (market.catalogue) and the streaming api at once:
            for runner_cata, runner in zip(market.market_catalogue.runners, market_book.runners):
                # Check the polling api and streaming api matches up (sometimes it doesn't)
                if runner_cata.selection_id == runner.selection_id:
                    # Get the dog_name from polling api then reference our data for our model rating
                    dog_name = runner_cata.runner_name.split(' ',1)[1].upper()

                    # Rest is the same as How to Automate III
                    model_price = todays_data.loc[dog_name,track,race_number]['pred_price'].item()
                    model_prob = todays_data.loc[dog_name,track,race_number]['conf'].item()
                    ### If you have an issue such as:
                        # Unknown error The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
                        # Then do model_price = todays_data.loc[dog_name,track,race_number]['rating'].item()

                    # Log info before placing bets
                    logging.info(f'dog_name: {dog_name}')
                    logging.info(f'model_price: {model_price}')
                    logging.info(f'market_id: {market_book.market_id}')
                    logging.info(f'selection_id: {runner.selection_id}')

                    delta_prob = ()
                    # If best available to back price is > rated price then flat $5 back
                    if runner.status == "ACTIVE" and runner.ex.available_to_back[0]['price'] > model_price and model_price>1 and runner.ex.available_to_back[0]['price'] < 30 :
                        #bf_price = runner.ex.available_to_back[0]['price']
                        implied_prob = 1/runner.ex.available_to_back[0]['price'] 
                        bet_amount = round((model_prob-implied_prob)*100*10,2)
                        logging.info(f"BF avail price: {runner.ex.available_to_back[0]['price']}")
                        logging.info(f'bet_amount: {bet_amount}')

                        trade = Trade(
                        market_id=market_book.market_id,
                        selection_id=runner.selection_id,
                        handicap=runner.handicap,
                        strategy=self,
                        )
                        order = trade.create_order(
                            side="BACK", order_type=LimitOrder(price=runner.ex.available_to_back[0]['price'], size=bet_amount, persistence_type='MARKET_ON_CLOSE')
                        )
                        market.place_order(order)
                    # If best available to lay price is < rated price then flat $5 lay



# logger = logging.getLogger(__name__)

"""
Worker can be used as followed:
    framework.add_worker(
        BackgroundWorker(
            framework,
            terminate,
            func_kwargs={"today_only": True, "seconds_closed": 1200},
            interval=60,
            start_delay=60,
        )
    )
This will run every 60s and will terminate 
the framework if all markets starting 'today' 
have been closed for at least 1200s
"""


# Function that stops automation running at the end of the day
def terminate(
    context: dict, flumine, today_only: bool = True, seconds_closed: int = 600
) -> None:
    """terminate framework if no markets
    live today.
    """
    markets = list(flumine.markets.markets.values())
    markets_today = [
        m
        for m in markets
        if m.market_start_datetime.date() == datetime.datetime.utcnow().date()
        and (
            m.elapsed_seconds_closed is None
            or (m.elapsed_seconds_closed and m.elapsed_seconds_closed < seconds_closed)
        )
    ]
    if today_only:
        market_count = len(markets_today)
    else:
        market_count = len(markets)
    if market_count == 0:
        # logger.info("No more markets available, terminating framework")
        flumine.handler_queue.put(TerminationEvent(flumine))

# Add the stopped to our framework






FIELDNAMES = [
    "bet_id",
    "strategy_name",
    "market_id",
    "selection_id",
    "trade_id",
    "date_time_placed",
    "price",
    "price_matched",
    "size",
    "size_matched",
    "profit",
    "side",
    "elapsed_seconds_executable",
    "order_status",
    "market_note",
    "trade_notes",
    "order_notes",
]


class LiveLoggingControl(LoggingControl):
    NAME = "BACKTEST_LOGGING_CONTROL"

    def __init__(self, *args, **kwargs):
        super(LiveLoggingControl, self).__init__(*args, **kwargs)
        self._setup()

    # Changed file path and checks if the file orders_hta_4.csv already exists, if it doens't then create it
    def _setup(self):
        if os.path.exists("orders_hta_4.csv"):
            logging.info("Results file exists")
        else:
            with open("orders_hta_4.csv", "w") as m:
                csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
                csv_writer.writeheader()

    def _process_cleared_orders_meta(self, event):
        orders = event.event
        with open("orders_hta_4.csv", "a") as m:
            for order in orders:
                if order.order_type.ORDER_TYPE == OrderTypes.LIMIT:
                    size = order.order_type.size
                else:
                    size = order.order_type.liability
                if order.order_type.ORDER_TYPE == OrderTypes.MARKET_ON_CLOSE:
                    price = None
                else:
                    price = order.order_type.price
                try:
                    order_data = {
                        "bet_id": order.bet_id,
                        "strategy_name": order.trade.strategy,
                        "market_id": order.market_id,
                        "selection_id": order.selection_id,
                        "trade_id": order.trade.id,
                        "date_time_placed": order.responses.date_time_placed,
                        "price": price,
                        "price_matched": order.average_price_matched,
                        "size": size,
                        "size_matched": order.size_matched,
                        "profit": 0 if not order.cleared_order else order.cleared_order.profit,
                        "side": order.side,
                        "elapsed_seconds_executable": order.elapsed_seconds_executable,
                        "order_status": order.status.value,
                        "market_note": order.trade.market_notes,
                        "trade_notes": order.trade.notes_str,
                        "order_notes": order.notes_str,
                    }
                    csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
                    csv_writer.writerow(order_data)
                except Exception as e:
                    logger.error(
                        "_process_cleared_orders_meta: %s" % e,
                        extra={"order": order, "error": e},
                    )

        logger.info("Orders updated", extra={"order_count": len(orders)})

    def _process_cleared_markets(self, event):
        cleared_markets = event.event
        for cleared_market in cleared_markets.orders:
            logger.info(
                "Cleared market",
                extra={
                    "market_id": cleared_market.market_id,
                    "bet_count": cleared_market.bet_count,
                    "profit": cleared_market.profit,
                    "commission": cleared_market.commission,
                },
            )



if __name__=="__main__":
    # Credentials to login and logging in 
    trading = betfairlightweight.APIClient('nickbarlow@live.com.au','76ff98a6',app_key='JFWqJHqB4Akfi5hK')
    client = clients.BetfairClient(trading, interactive_login=True)

    # Login
    framework = Flumine(client=client)

    # Code to login when using security certificates
    # trading = betfairlightweight.APIClient('username','password',app_key='appkey', certs=r'C:\Users\zhoui\openssl_certs')
    # client = clients.BetfairClient(trading)
    # framework = Flumine(client=client)
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    logging.basicConfig(filename = f'./logs/fluminelog {today}.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    todays_data = pd.read_pickle(f'./inputs/output VIC {today}.npy')

    todays_data['DogName_bf'] = todays_data['dogs'].apply(lambda x: x.replace("'", "").replace(".", "").replace("Res", "").strip())
    todays_data.replace('Sandown (SAP)','Sandown Park',inplace = True)
    todays_data.replace('Christchurch (NZ)','Addington',inplace = True)
    todays_data.replace('Palmerston Nth (NZ)','Manawatu',inplace = True)
    todays_data.replace('Waikato','Cambridge',inplace = True)
    todays_data.replace('Wagga Wagga','Wagga',inplace = True)
    tracks = todays_data.track.unique().tolist()
    todays_data =todays_data.set_index(['DogName_bf','track','race_num'])
    print(tracks)


    logger = logging.getLogger(__name__)

    framework.add_logging_control(
        LiveLoggingControl()
    )

    framework.add_worker(
    BackgroundWorker(
        framework,
        terminate,
        func_kwargs={"today_only": True, "seconds_closed": 1200},
        interval=60,
        start_delay=60,
    )
    )

    greyhounds_strategy = FlatBetting(
        market_filter=streaming_market_filter(
            event_type_ids=["4339"], # Greyhounds markets
            country_codes=["AU"], # Australian markets
            market_types=["WIN"], # Win markets
        ),
        max_selection_exposure=500,
        max_order_exposure= 500, # Max exposure per order = 50
        max_trade_count=1, # Max 1 trade per selection
        max_live_trade_count=1, # Max 1 unmatched trade per selection
    )

    framework.add_strategy(greyhounds_strategy)

    framework.run()