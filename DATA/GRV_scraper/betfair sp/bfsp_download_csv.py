from datetime import date, timedelta
import pickle
import requests_async as requests
from tqdm import tqdm

import asyncio
import time


async def do_work(link: str, delay_s: float = 1.0):
    
    fname = "./DATA/bfsp_csv/"+link[-12:]
    print(f"{fname} started")
    response = await requests.get(link)
    open(fname, "wb").write(response.content)
    # await asyncio.sleep(delay_s)
    print(f"{fname} done")

async def worker(self):
    while True:
        try:
            await self.process_one()
        except asyncio.CancelledError:
            return

async def process_one(self):
    url = await self.todo.get()
    try:
        await self.crawl(url)
    except Exception as exc:
        # retry handling here...
        pass
    finally:
        self.todo.task_done()

async def main():
    start_date = date(2023, 2, 1) 
    end_date = date.today()   # perhaps date.now()

    delta = end_date - start_date   # returns timedelta

    day_list = []

    for task in done:
        result = task.result()
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        print(day.strftime('%d%m%y'))
        day_list.append(f"https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin{day.strftime('%d%m%Y')}.csv")
    day_list


    tasks = [asyncio.create_task(do_work(item)) for item in day_list]
    done, pending = await asyncio.wait(tasks)

    # for link in tqdm(day_list):
    #     fname = "./DATA/bfsp_csv/"+link[-12:]
    #     response = requests.get(link)
    #     open(fname, "wb").write(response.content)

if __name__=="__main__":
    asyncio.run(main())
