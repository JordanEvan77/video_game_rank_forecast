from setup.setups import api_key
from setup.setups import headers
from setup.setups import dir_base
import requests
import pandas as pd
import numpy as np
import time
import random

tiers = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER']  # All tiers
match_details = []

# Test API
url = 'https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/BRONZE/I?page=1'
response = requests.get(url, headers=headers)
print(response)

overall_start_time = time.time()

for tier in tiers:
    print(f'***** tier:{tier}, time:{(time.time() - overall_start_time) / 60:.2f} minutes *****')
    match_details = []
    summoners_all_pages = []

    page = 1
    while True:
        url_leaderboard_page = f"https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/I?page={page}"
        response_leaderboard_page = requests.get(url_leaderboard_page, headers=headers)
        summoners_page = response_leaderboard_page.json()

        if not summoners_page:  # empty page is break
            break

        summoners_all_pages.extend(summoners_page)
        page += 1
        print(f"Fetched page {page} for {tier}, time:{(time.time() - overall_start_time) / 60:.2f} minutes")
        time.sleep(1)

    sampled_summoners = random.sample(summoners_all_pages, min(len(summoners_all_pages), 10_000))

    i = 0
    for summoner in sampled_summoners:
        i += 1
        print(f"{i} out of {len(sampled_summoners)}, time:{(time.time() - overall_start_time) / 60:.2f} minutes")
        summoner_id = summoner['summonerId']
        url_puuid = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
        response_puuid = requests.get(url_puuid, headers=headers)
        puuid = response_puuid.json()['puuid']

        start_date = int(time.mktime(time.strptime("2023-01-01", "%Y-%m-%d")))
        end_date = int(time.mktime(time.strptime("2024-12-31", "%Y-%m-%d")))

        url_matchlist = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={start_date}&endTime={end_date}"
        response_matchlist = requests.get(url_matchlist, headers=headers)
        match_ids = response_matchlist.json()

        url_match = f"https://americas.api.riotgames.com/lol/match/v5/matches/"
        for match_id in match_ids:
            response_match = requests.get(f"{url_match}{match_id}", headers=headers)
            match_details.append(response_match.json())
            time.sleep(1)

    # save out!
    match_df = pd.DataFrame(match_details)
    parquet_filename = dir_base + f"data/class_raw_data/{tier.lower()}_match_details.parquet"
    match_df.to_parquet(parquet_filename)
    print(f"Saved {parquet_filename}, time:{(time.time() - overall_start_time) / 60:.2f} minutes")
