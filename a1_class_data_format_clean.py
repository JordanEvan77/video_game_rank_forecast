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
from video_game_rank_forecast.a0_reg_class_api_call import get_common_columns, \
    read_with_common_columns, tiers_list
#TODO: Make sure no time series factor is impactful

# I will kind of have two different datasets, for the different tasks
#Regression tasks: Overall tier/LP?
#Classification Tasks: Game to game victory, final tier placement

# lets start with the binary task of game to game victory.

def flatten_nest_dict(df_dict):
    '''
    a simple recursive funciton to help with the terrible nested dictionaries
    :param df_dict:
    :return:
    '''
    items, sep, parent_key = [], '_', ''
    for k, v in df_dict.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nest_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)





#Exploratory Analysis
def early_eda(start_df):
    # How many nested columns are there?
    print(start_df.shape) #(145,217, 3) less than a million rows, thats good!
    print(start_df.head(5))

    # I have metadata, and info as dictionaries, check both
    # INFO:
    info_temp = start_df.iloc[1,0]
    print('list of keys in info', info_temp.keys())
    #list of keys in info dict_keys(['dataVersion', 'matchId', 'participants'])
    # these aren't needed? since we already have summoner ID


    meta_temp = start_df.iloc[1,1]
    print('list of keys in meta', meta_temp.keys())
    # Now this is the complicated part
    #list of keys in meta dict_keys(['endOfGameResult', 'gameCreation', 'gameDuration', 'gameEndTimestamp', 'gameId', 'gameMode', 'gameName', 'gameStartTimestamp', 'gameType', 'gameVersion', 'mapId', 'participants', 'platformId', 'queueId', 'teams', 'tournamentCode'])

    # 'gameDuration' is interesting, and 'win' is my target
    start_df = start_df.drop('info', axis=0)

    flat_df = pd.json_normalize(start_df['metadata'].apply(flatten_nest_dict)) # apply the recursion

    raw_df = pd.concat([start_df.drop(columns=['info']), flat_df], axis=1)

    #now check columns again:
    print(raw_df.columns)

    start_df['matchId'] = start_df.metadata['matchId']
    start_df['gameDuration'] = start_df.metadata['matchId']




    #TODO: This match is attached to a single summoners ID. I should keep it that way, and drop 
    # the info about all other summoners, as they will have their own match version of the same 
    # match, and I don't want to 10x my data with duplicates for no reason.  This also means that
    # at least this point we don't care as much about the other teams info



    # which ones seem interesting? check correlation

    #which ones are too obvious, and have to do with data leak?

    #which ones are most likely associated with early game decisions?
    win_con = ['win']
    columns = [
        "12AssistStreakCount",
        "HealFromMapSmyces",
        "SWARM_DefeatMiniBosses",
        "controlWardTimeCoverageInRiverOrEnemyHalf",
        "dodgeSkillShotsSmallWindow",
        "earlyLaningPhaseGoldExpAdvantage",
        "effectiveHealAndShielding",
        "enemyChampionImmobilizations",
        "firstTurretKilledTime",
        "goldPerMinute",
        "immobilizeAndKillWithAlly",
        "jungleCsBefore10Minutes",
        "killParticipation",
        "landSkillShotsEarlyGame",
        "moreEnemyJungleThanOpponent",
        "outerTurretExecutesBefore10Minutes",
        "teamDamagePercentage",
        "visionScorePerMinute",
        "wardTakedownsBefore20M",
        "assists",
        "baronKills",
        "deaths",
        "damageDealtToObjectives",
        "damageDealtToTurrets",
        "damageSelfMitigated",
        "dragonKills",
        "firstBloodAssist",
        "inhibitorTakedowns",
        "neutralMinionsKilled",
        "physicalDamageDealt",
        "totalDamageDealt",
        "totalHeal",
        "totalTimeCCDealt",
        "trueDamageDealt",
        "visionScore",
        "objectives",
        "teamDamagePercentage",
        "teamRiftHeraldKills",
        "initialBuffCount",
        "jungleCsBefore10Minutes",
        "initialCrabCount",
        "takedownsFirstXMinutes",
        "killsOnLanersEarlyJungleAsJungler",
        "killsOnOtherLanesEarlyJungleAsLaner",
        "earlyLaningPhaseGoldExpAdvantage",
        "laneMinionsFirst10Minutes",
        "controlWardsPlaced",
        "visionScorePerMinute",
        "firstTurretKilledTime",
        "turretPlatesTaken",
        "abilityUses",
        "landSkillShotsEarlyGame"
    ]

#Categorical Cleaning and Encoding
# Encode final Ranking


#Numeric Cleaning



def complex_read_in(parquet_high_name, tiers_list, common_columns):
    parquet_files = []
    df_list = []
    for tier in tiers_list:
        tier_file = parquet_high_name + f"//{tier}"
        for parquet in os.listdir(tier_file):
            print(tier, parquet)
            df = pd.read_parquet(tier_file + f"/{parquet}", columns=common_columns)
            df_list.append(df)
    return pd.concat(df_list, axis=0)


if __name__ == '__main__':
    print('start!')
    start_time = time.time()
    past_run_date = '01-01-2025'

    parquet_high_name = dir_base + f"data/class_raw_data_{past_run_date}"

    common_columns = ['metadata', 'info', 'summoner_id']
    start_df = complex_read_in(parquet_high_name, tiers_list, common_columns)
    print('read in complete', (time.time() - start_time) / 60)

    early_eda(start_df)