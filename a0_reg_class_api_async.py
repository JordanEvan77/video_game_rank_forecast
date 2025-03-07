from setup.setups import api_key, headers, dir_base
import aiohttp
import asyncio
import requests
import pandas as pd
import numpy as np
import time
import random
import datetime
import os
import pyarrow.parquet as pq
import pyarrow as pa
tiers_list = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'DIAMOND']
overall_start_time = time.time()

async def fetch(session, url, headers):
    start_time = time.time()
    async with session.get(url, headers=headers) as response:
        response_data = await response.json()
        #print('fetch complete')
        return response_data

async def get_total_pages(session, url, headers):
    async with session.get(url, headers=headers) as response:
        response_data = await response.json()
        print('total pages')
        return len(response_data)



async def get_summoners_for_tier(tier):
    page = 1
    all_summoners = []
    async with aiohttp.ClientSession() as session:
        while True:
            url = f"https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/I?page={page}"
            summoners = await fetch(session, url, headers)
            if not summoners:
                break
            all_summoners.extend(summoners)
            page += 1
    return all_summoners


async def get_summoner_list(force): #sample_size,
    final_sampled_df = pd.DataFrame()
    sample_id_path = dir_base + "data/all_tier_summoner_ids.csv"
    if force != True and os.path.exists(sample_id_path):
        return pd.read_csv(sample_id_path)
    for tier in tiers_list: # iterate through tiers with wait
        print(f'***** tier:{tier}, time:{(time.time() - overall_start_time) / 60} minutes *****')
        summoners_all_pages = await get_summoners_for_tier(tier)
        #need to remove odd strings!
        summoners_all_pages = [item for item in summoners_all_pages if isinstance(item, dict)]
        sampled_df = pd.DataFrame(summoners_all_pages) # was cuasing error
        final_sampled_df = pd.concat([final_sampled_df, sampled_df])

    final_cols = [i for i in final_sampled_df.columns if i != 'leagueId']
    final_sampled_df = final_sampled_df[final_cols]
    final_sampled_df = final_sampled_df.rename(columns={'summonerId':'summoner_id', 'leaguePoints':'LP'})
    final_sampled_df.to_csv(sample_id_path, index=False)
    return final_sampled_df


def get_common_columns(files):
    common_columns = None
    for filepath in files:
        df = pd.read_parquet(filepath)
        if common_columns is None:
            common_columns = set(df.columns)
        else:
            common_columns.intersection_update(df.columns)
    return list(common_columns)


def read_with_common_columns(filepath, common_columns):
    df = pd.read_parquet(filepath, columns=common_columns)
    return df


async def get_match_details(sampled_df, run_date, force=True):
    os.makedirs(dir_base + f"data/class_raw_data_{run_date}", exist_ok=True)
    async with aiohttp.ClientSession() as session:
        for tier in tiers_list:
            os.makedirs(dir_base + f"data/class_raw_data_{run_date}/{tier.lower()}", exist_ok=True)
            min_id, max_id = 0, 100
            sampled_df_tier = sampled_df[sampled_df['tier'] == tier]

            sampled_df_tier_list = list(sampled_df_tier.summoner_id.unique())

            match_details = []
            tasks = []
            parquet_filename = dir_base + f"data/class_raw_data_{run_date}/{tier.lower()}/" \
                                          f"{tier.lower()}_match_details_{run_date}_{min_id}_" \
                                          f"{max_id}.parquet" #before loop
            for i, summoner in enumerate(sampled_df_tier_list):
                print(f"Grabbing tier: {tier}, {i + 1} out of {sampled_df_tier.shape[0]}, time"
                      f":{(time.time() - overall_start_time) / 60:.2f} minutes")
                summoner_id = summoner
                url_puuid = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
                tasks.append(fetch(session, url_puuid, headers))
                time.sleep(2)  # The api can handle 20 requests per second and 50 per minute
                if i % 20 == 0: # trigger async for every 50 players
                    responses_puuid = await asyncio.gather(*tasks)
                    tasks = []
                    for response_puuid in responses_puuid:
                        try:
                            puuid = response_puuid['puuid']
                            start_date = int(time.mktime(time.strptime("2024-01-01", "%Y-%m-%d")))
                            end_date = int(time.mktime(time.strptime("2024-12-31", "%Y-%m-%d")))
                            url_matchlist = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={start_date}&endTime={end_date}"
                            tasks.append(fetch(session, url_matchlist, headers))
                            time.sleep(2)
                        except KeyError:
                            continue
                if i % 20 == 0:
                    match_ids_responses = await asyncio.gather(*tasks)
                    tasks = []
                    for match_ids in match_ids_responses:
                        for match_id in match_ids:
                            url_match = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
                            tasks.append(fetch(session, url_match, headers))
                            time.sleep(2)
                        # no threshold, some folks may not play 50 games ina season
                        responses_match = await asyncio.gather(*tasks)
                        tasks = []
                        for response_match in responses_match:
                            if isinstance(response_match, dict):
                                dict_temp_match = response_match
                                dict_temp_match['summoner_id'] = summoner_id
                                match_details.append(dict_temp_match)

                if i % 100 == 0 and match_details:
                    match_df = pd.DataFrame(match_details)
                    match_df.to_parquet(parquet_filename)
                    print(f"Saved {parquet_filename}, time:{(time.time() - overall_start_time) / 60:.2f} minutes")
                    min_id = i
                    max_id = i + 100
                    parquet_filename = dir_base + f"data/class_raw_data_{run_date}/{tier.lower()}/" \
                                                  f"{tier.lower()}_match_details_{run_date}_{min_id}_{max_id}.parquet"
                    match_details = []

            if tasks: # gather the tail end of items
                responses_match = await asyncio.gather(*tasks)
                for response_match in responses_match:
                    if isinstance(response_match, dict):
                        dict_temp_match = response_match
                        dict_temp_match['summoner_id'] = summoner_id
                        match_details.append(dict_temp_match)

            if match_details: # final tail end save out
                match_df = pd.DataFrame(match_details)
                match_df.to_parquet(parquet_filename)
                print(f"Final save {parquet_filename}, time:{(time.time() - overall_start_time) / 60:.2f} minutes")


if __name__ == '__main__':
    # Test API
    url = 'https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/BRONZE/I?page=1'
    response = requests.get(url, headers=headers)
    print(response)
    if str(response) == '<Response [403]>':
        print('BAD RESPONSE')
        raise ValueError("Invalid value provided!")

    print('start!')
    loop = asyncio.get_event_loop()
    sampled_df = loop.run_until_complete(get_summoner_list(force=False))
    run_date = datetime.datetime.today().strftime("%m-%d-%Y")
    loop.run_until_complete(get_match_details(sampled_df, run_date, force=False))


## FULL RUN SUCCESS ON 3/6 with ALL items, no further sampling needed!

#THis async is more of a proof of concept, it cleans up the code base, and allows for a much
# higher rate of retrieval. The API can only handle 50 requests a minute, so this async process
# naturally runs fater than the API can handle, but in a business case, witha  stronger API this
# approach would be useful.