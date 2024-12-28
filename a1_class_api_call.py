#Initial adding of league of legends api


#'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER']  # All tiers

from setup.setups import api_key
from setup.setups import headers
from setup.setups import dir_base
import requests
import pandas as pd
import numpy as np
import time
import random

tiers = ['IRON','BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER']  # All tiers
match_details = []

# Test API
url = 'https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/BRONZE/I?page=1'
response = requests.get(url, headers=headers)
print(response)

start_time = time.time()

for tier in tiers:
    print(f'tier:{tier}, time:{(time.time()-start_time)/60}')
    match_details = []  # one set per tier
    url_leaderboard_first_page = f"https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/I?page=1"
    response_leaderboard_first_page = requests.get(url_leaderboard_first_page, headers=headers)
    summoners_first_page = response_leaderboard_first_page.json()

    url_leaderboard_last_page = f"https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/I?page=100"  # Adjust if there are fewer pages
    response_leaderboard_last_page = requests.get(url_leaderboard_last_page, headers=headers)
    summoners_last_page = response_leaderboard_last_page.json()

    sampled_summoners_first_page = random.sample(summoners_first_page, min(len(summoners_first_page), 5))
    sampled_summoners_last_page = random.sample(summoners_last_page, min(len(summoners_last_page), 5))

    sampled_summoners = sampled_summoners_first_page + sampled_summoners_last_page

    i = 0
    for summoner in sampled_summoners:
        i += 1
        print(f"{i} out of {len(sampled_summoners)}, time:{(time.time()-start_time)/60}")
        summoner_id = summoner['summonerId']
        url_puuid = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
        response_puuid = requests.get(url_puuid, headers=headers)
        puuid = response_puuid.json()['puuid']

        start_date = int(time.mktime(time.strptime("2023-01-01", "%Y-%m-%d")))
        end_date = int(time.mktime(time.strptime("2024-12-31", "%Y-%m-%d")))

        url_matchlist = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/" \
                        f"{puuid}/ids?startTime={start_date}&endTime={end_date}"
        response_matchlist = requests.get(url_matchlist, headers=headers)
        match_ids = response_matchlist.json()

        url_match = f"https://americas.api.riotgames.com/lol/match/v5/matches/"
        for match_id in match_ids:
            response_match = requests.get(f"{url_match}{match_id}", headers=headers)
            match_details.append(response_match.json())
            time.sleep(1)

    # save as df parquet
    match_df = pd.DataFrame(match_details)
    parquet_filename = dir_base + f"data\class_raw_data\{tier.lower()}_match_details.parquet"
    match_df.to_parquet(parquet_filename)
    print(f"Saved {parquet_filename}, time:{(time.time()-start_time)/60}")

