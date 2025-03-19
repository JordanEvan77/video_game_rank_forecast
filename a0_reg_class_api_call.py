from setup.setups import api_key
from setup.setups import headers
from setup.setups import dir_base
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

def get_summoner_list(sample_size, force):
    # check if list exists and has same size
    final_smapled_df = pd.DataFrame()
    sample_id_path = dir_base + f"data/all_tier_summoner_id_size_{sample_size}.csv"
    if os.path.exists(sample_id_path) and force != True:
        sampled_df = pd.read_csv(sample_id_path)
        return sampled_df

    #if it doesn't exist, run the search
    for tier in tiers_list:
        #TODO: Since key is 24 hr, may need to save summoner tags somehow? And pick up where we left
        # off?
        print(f'***** tier:{tier}, time:{(time.time() - overall_start_time) / 60:.2f} minutes *****')
        summoners_all_pages = []

        page = 1
        while True:
            url_leaderboard_page = f"https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/I?page={page}"
            response_leaderboard_page = requests.get(url_leaderboard_page, headers=headers)
            if str(response_leaderboard_page) == '<Response [400]>':
                raise ValueError("Invalid value provided!")

            summoners_page = response_leaderboard_page.json()

            if not summoners_page:  # empty page is break
                break

            summoners_all_pages.extend(summoners_page)
            page += 1
            if page % 100 == 0:
                print(f"Fetched page {page} for {tier}, time:{(time.time() - overall_start_time) / 60:.2f} minutes")
            time.sleep(1)

            #TODO: use a temporary break, remove:
            #if page == sample_size:
                #break
        if sample_size =='max':
            sampled_summoners = summoners_all_pages
        else:
            sampled_summoners = random.sample(summoners_all_pages, min(len(summoners_all_pages),
                                                                   sample_size))
        sampled_df = pd.DataFrame(sampled_summoners)
        final_smapled_df = pd.concat([final_smapled_df, sampled_df])

    final_cols = [i for i in final_smapled_df.columns if i != 'leagueId']
    final_smapled_df = final_smapled_df[final_cols]
    final_smapled_df = final_smapled_df.rename(columns={'summonerId':'summoner_id',
                                                       'leaguePoints':'LP'})
    final_smapled_df.to_csv(sample_id_path, index=False)
    return final_smapled_df


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
    if force != True:
        print('read in previous version')
        parquet_files = [parquet_high_name + '\\' + f for f in os.listdir(parquet_high_name) if
                         f.endswith(".parquet")]
        # if this is empty then we need to go to start of reading in!
        if parquet_files == []:
            print('Empty tier, just start reading')
            sampled_df_tier_list = list(sampled_df_tier.summoner_id.unique())
            return sampled_df_tier_list, sampled_df_tier, min_id, max_id
        common_columns = get_common_columns(parquet_files)
        aligned_dfs = []
        #The above needs to be allowed to be empty, so that it can skip the following:

        print('got common columns')
        for filepath in parquet_files:
            df = read_with_common_columns(filepath, common_columns)
            aligned_dfs.append(df)

        combined_df = pd.concat(aligned_dfs, axis=0)

        old_summoners_id = combined_df.summoner_id.unique()
        print('already saved ids read in:', len(old_summoners_id))
        min_id += len(old_summoners_id)
        max_id += len(old_summoners_id)
        sampled_df_tier = sampled_df_tier[~sampled_df_tier['summoner_id'].isin(
            old_summoners_id)]

    sampled_df_tier_list = list(sampled_df_tier.summoner_id.unique())
    return sampled_df_tier_list, sampled_df_tier, min_id, max_id


def get_match_details(sampled_df, run_date, force=True):
    old_check = 'n'
    os.makedirs(dir_base + f"data/class_raw_data_{run_date}", exist_ok=True)
    for tier in tiers_list:
        os.makedirs(dir_base + f"data/class_raw_data_{run_date}/{tier.lower()}", exist_ok=True)
        min_id, max_id = 0, 100
        # get reduced list
        sampled_df_tier = sampled_df[sampled_df['tier'] == tier]

        parquet_filename = dir_base + f"data/class_raw_data_{run_date}/{tier.lower()}/" \
        f"{tier.lower()}_match_details_{run_date}_{min_id}_{max_id}.parquet"
        parquet_high_name = dir_base + f"data/class_raw_data_{run_date}/{tier.lower()}" # should

        sampled_df_tier_list, sampled_df_tier, min_id, max_id = get_sample_ids(parquet_high_name,
                                                   sampled_df_tier, force, min_id, max_id)

        match_details = [] # needs to set up here, and then reset every time it saves out
        # so that the computation doesn't suffer
        i = 0
        for summoner in sampled_df_tier_list:
            i += 1
            print(f"Grabbing tier: {tier}, {i} out of {sampled_df_tier.shape[0]}, time"
                  f":{(time.time() - overall_start_time) / 60:.2f} minutes")
            summoner_id = summoner
            url_puuid = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
            response_puuid = requests.get(url_puuid, headers=headers)
            try:
                puuid = response_puuid.json()['puuid'] # fails here
            except: continue # pass on fails

            start_date = int(time.mktime(time.strptime("2023-01-01", "%Y-%m-%d")))
            end_date = int(time.mktime(time.strptime("2024-12-31", "%Y-%m-%d")))

            print('grabbing match list')
            time.sleep(1)
            url_matchlist = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={start_date}&endTime={end_date}"
            response_matchlist = requests.get(url_matchlist, headers=headers)
            match_ids = response_matchlist.json()

            url_match = f"https://americas.api.riotgames.com/lol/match/v5/matches/"
            print('checking matches')
            for match_id in match_ids:
                #print(match_id)
                try:
                    response_match = requests.get(f"{url_match}{match_id}", headers=headers)
                    dict_temp_match = response_match.json()
                    dict_temp_match['summoner_id'] = summoner_id
                    match_details.append(dict_temp_match) # include summonerid

                #some failures, catch below
                except:
                    print('Didnt connect')
                    time.sleep(10)
                    try:
                        response_match = requests.get(f"{url_match}{match_id}", headers=headers)
                        match_details.append(response_match.json())
                    except:
                        match_details.append([np.nan])# does this have to be shaped different?
                time.sleep(1)


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
    # , 'MASTER', 'GRANDMASTER', 'CHALLENGER']  # All tiers

    # Test API
    url = 'https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/BRONZE/I?page=1'
    response = requests.get(url, headers=headers)
    print(response)
    if str(response) == '<Response [403]>':
        print('BAD RESPONSE')
        raise ValueError("Invalid value provided!")


    print('start!')
    sampled_df = get_summoner_list(sample_size='max', force=False)
    run_date = datetime.datetime.today().strftime("%m-%d-%Y")
    run_date = "03-14-2025"
    get_match_details(sampled_df, run_date, force=False)