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
#, 'MASTER', 'GRANDMASTER', 'CHALLENGER']  # All tiers

# Test API
url = 'https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/BRONZE/I?page=1'
response = requests.get(url, headers=headers)
print(response)
if str(response) == '<Response [403]>':
    print('BAD RESPONSE')
    raise ValueError("Invalid value provided!")

overall_start_time = time.time()

async def fetch(session, url, headers):
    async with session.get(url, headers=headers) as response:
        return await response.json()

async def get_summoners_for_tier(tier, sample_size):
    urls = [f"https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/I?page={page}"
            for page in range(1, sample_size//100 + 2)]
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(fetch(session, url, headers))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
        summoners_all_pages = [summoner for response in responses for summoner in response]
        return summoners_all_pages

async def get_summoner_list(sample_size, force):
    # check if list exists and has same size
    final_sampled_df = pd.DataFrame()
    sample_id_path = dir_base + f"data/all_tier_summoner_id_size_{sample_size}.csv"
    if os.path.exists(sample_id_path) and not force:
        sampled_df = pd.read_csv(sample_id_path)
        return sampled_df

    #if it doesn't exist, run the search
    for tier in tiers_list:
        #TODO: Since key is 24 hr, may need to save summoner tags somehow? And pick up where we left off?
        print(f'***** tier:{tier}, time:{(time.time() - overall_start_time) / 60:.2f} minutes *****')
        summoners_all_pages = await get_summoners_for_tier(tier, sample_size)
        sampled_summoners = random.sample(summoners_all_pages, min(len(summoners_all_pages), sample_size))
        sampled_df = pd.DataFrame(sampled_summoners)
        final_sampled_df = pd.concat([final_sampled_df, sampled_df])

    final_cols = [i for i in final_sampled_df.columns if i != 'leagueId']
    final_sampled_df = final_sampled_df[final_cols]
    final_sampled_df = final_sampled_df.rename(columns={'summonerId':'summoner_id', 'leaguePoints':'LP'})
    final_sampled_df.to_csv(sample_id_path, index=False)
    return final_sampled_df

# I need functions to help clean up all the dictionary stuff, as the formats are clashing:
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

def get_sample_ids(parquet_high_name, sampled_df_tier, force, min_id, max_id):
    '''
    A function that reads in the previous effort to check for already saved data
    :param parquet_high_name: file path name for tier
    :param sampled_df_tier: original set of IDs
    :param force: trigger for fresh start
    :param min_id: string identifier for save out
    :param max_id: string identifier for save out
    :return: set of IDs to iterate through
    '''
    # check in whole set of previously read in
    if not force:
        print('read in previous version')
        parquet_files = [parquet_high_name + '\\' + f for f in os.listdir(parquet_high_name) if f.endswith(".parquet")]
        # if this is empty then we need to go to start of reading in!
        if not parquet_files:
            print('Empty tier, just start reading')
            sampled_df_tier_list = list(sampled_df_tier.summoner_id.unique())
            return sampled_df_tier_list, sampled_df_tier, min_id, max_id
        common_columns = get_common_columns(parquet_files)
        aligned_dfs = []
        # The above needs to be allowed to be empty, so that it can skip the following:

        print('got common columns')
        for filepath in parquet_files:
            df = read_with_common_columns(filepath, common_columns)
            aligned_dfs.append(df)

        combined_df = pd.concat(aligned_dfs, axis=0)

        old_summoners_id = combined_df.summoner_id.unique()
        print('already saved ids read in:', len(old_summoners_id))
        min_id += len(old_summoners_id)
        max_id += len(old_summoners_id)
        sampled_df_tier = sampled_df_tier[~sampled_df_tier['summoner_id'].isin(old_summoners_id)]

    sampled_df_tier_list = list(sampled_df_tier.summoner_id.unique())
    return sampled_df_tier_list, sampled_df_tier, min_id, max_id

async def get_match_details(sampled_df, run_date, force=True):
    os.makedirs(dir_base + f"data/class_raw_data_{run_date}", exist_ok=True)
    async with aiohttp.ClientSession() as session:
        for tier in tiers_list:
            os.makedirs(dir_base + f"data/class_raw_data_{run_date}/{tier.lower()}", exist_ok=True)
            min_id, max_id = 0, 100
            # get reduced list
            sampled_df_tier = sampled_df[sampled_df['tier'] == tier]

            parquet_filename = dir_base + f"data/class_raw_data_{run_date}/{tier.lower()}/" \
                                          f"{tier.lower()}_match_details_{run_date}_{min_id}_{max_id}.parquet"
            parquet_high_name = dir_base + f"data/class_raw_data_{run_date}/{tier.lower()}"

            sampled_df_tier_list, sampled_df_tier, min_id, max_id = get_sample_ids(parquet_high_name,
                                                                                   sampled_df_tier, force, min_id, max_id)
            match_details = []
            for i, summoner in enumerate(sampled_df_tier_list):
                print(f"Grabbing tier: {tier}, {i + 1} out of {sampled_df_tier.shape[0]}, time"
                      f":{(time.time() - overall_start_time) / 60:.2f} minutes")
                summoner_id = summoner
                url_puuid = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
                response_puuid = await fetch(session, url_puuid, headers)
                try:
                    puuid = response_puuid['puuid']
                except KeyError:
                    continue

                start_date = int(time.mktime(time.strptime("2023-01-01", "%Y-%m-%d")))
                end_date = int(time.mktime(time.strptime("2024-12-31", "%Y-%m-%d")))

                print('grabbing match list')
                await asyncio.sleep(1)
                url_matchlist = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={start_date}&endTime={end_date}"
                match_ids = await fetch(session, url_matchlist, headers)

                url_match = f"https://americas.api.riotgames.com/lol/match/v5/matches/"
                print('checking matches')
                for match_id in match_ids:
                    try:
                        response_match = await fetch(session, f"{url_match}{match_id}", headers)
                        dict_temp_match = response_match
                        dict_temp_match['summoner_id'] = summoner_id
                        match_details.append(dict_temp_match) # include summonerid

                    #some failures, catch below
                    except:
                        print('Didnt connect')
                        await asyncio.sleep(10)
                        try:
                            response_match = await fetch(session, f"{url_match}{match_id}", headers)
                            match_details.append(response_match)
                        except:
                            match_details.append([np.nan]) # does this have to be shaped different?
                    await asyncio.sleep(1)

                # save out!
            if i % 100 ==0:
                print('stacking DF for summoner')
                match_df = pd.DataFrame(match_details)
                match_df.to_parquet(parquet_filename)
                print(f"Saved {parquet_filename}, time:{(time.time() - overall_start_time) / 60:.2f} minutes")
                min_id += 100
                max_id += 100
                parquet_filename = dir_base + f"data/class_raw_data_{run_date}/{tier.lower()}/" \
                                              f"{tier.lower()}_match_details_{run_date}_{min_id}_{max_id}.parquet"
                match_details = [] # reset for next batch


if __name__ == '__main__':
    print('start!')
    loop = asyncio.get_event_loop()
    sampled_df = loop.run_until_complete(get_summoner_list(sample_size=2000, force=False))
    run_date = datetime.datetime.today().strftime("%m-%d-%Y")
    #run_date = "01-30-2025"
    loop.run_until_complete(get_match_details(sampled_df, run_date, force=False))