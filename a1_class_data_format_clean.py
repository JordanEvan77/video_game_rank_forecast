from setup.setups import api_key
from setup.setups import headers
from setup.setups import dir_base
import requests
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import os
import pyarrow.parquet as pq
import pyarrow as pa
from video_game_rank_forecast.a0_reg_class_api_call import tiers_list
pd.options.mode.chained_assignment = None
#TODO: Make sure no time series factor is impactful
iter = 0
# I will kind of have two different datasets, for the different tasks
#Regression tasks: Overall tier/LP?
#Classification Tasks: Game to game victory, final tier placement

# lets start with the binary task of game to game victory.

def key_col_holder():
    key_columns = ['win', 'summonerId',
        "kills", # this is extremely obvious and correlated, still want to check on it though
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
        "landSkillShotsEarlyGame",
        'allInPings',
        'assistMePings',
        'baitPings',
        'basicPings',
        'bountyLevel',
        'teamPosition',
        'timeCCingOthers',
        'timePlayed',
        'totalAllyJungleMinionsKilled',
        'totalDamageDealtToChampions',
        'totalDamageShieldedOnTeammates',
        'totalDamageTaken',
        'totalEnemyJungleMinionsKilled',
        'totalHealsOnTeammates',
        'totalMinionsKilled',
        'totalTimeSpentDead',
        'totalUnitsHealed',
        'trueDamageDealtToChampions',
        'trueDamageTaken',
        'turretKills',
        'turretTakedowns',
        'turretsLost',
        'unrealKills',
        'visionClearedPings',
        'visionWardsBoughtInGame',
        'wardsKilled',
        'wardsPlaced'
    ]
    #Ignoring all challenges columns, as they conflate with other columns already selected
    return key_columns


def flatten_nest_dict(df_dict, parent_key='', sep='_'):
    '''
    A complex recursive function to unpack the dictionaries
    :param df_dict: the item to iterate and flatten
    :param parent_key: prior key
    :param sep: separate in new column name
    :return:
    '''

    items = []
    #print('starter', df_dict)
    if df_dict is None:
        #print('return empty') # base case return
        return {}
    if isinstance(df_dict, dict): # first lets check for standard dictionaries
        #print('***', df_dict)
        for k, v in df_dict.items(): # pull out key and value, see if the value is still a
            # dictionary
            #print('<<<' * 55)
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, (dict, list, np.ndarray)): # if the item within the dictionary is is
                # a dictionary, list or array, make another recursive call
                items.extend(flatten_nest_dict(v, new_key, sep=sep).items())
            else:
                #print('opt out') # now catches plain ints and sends them back up the recursion
                items.append((new_key, v))

    elif isinstance(df_dict, (list, np.ndarray)): # special handling of arrays and lists,
        # so that it can enumerate
        #print(len(df_dict))
        for i, v in enumerate(df_dict):
            #print(i, v)
            new_key = f'{parent_key}{sep}{i}' if parent_key else str(i)
            if isinstance(v, (dict, list, np.ndarray)):
                items.extend(flatten_nest_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
    else:
        #print('simple append')
        items.append((parent_key, df_dict))

    return dict(items)
# the big lesson learned with this function is making sure to start with the simplest items first
# and then add complexity. It was also interesting learning that I could very easily handle
# arrays, lists and dictionaries in the same recursion call, with some tweaking of the checks.


def find_participant_number(row, col_list):
    '''
    A simple function that can be applied to find out which participant each summoner is in their own match.
    :param col_list: list of id columns
    :param row: individual row to check
    :return: participant number
    '''

    for col in col_list:
        if row['summoner_id'] == row[col]:
            return col.split('_')[1]
    return None


def flatten_and_reduce_df(start_df, start_time):
    # I have metadata, and info as dictionaries, check both
    # INFO:
    meta_temp = start_df.iloc[1, 0]
    #print('list of keys in meta', meta_temp.keys())
    # list of keys in info dict_keys(['dataVersion', 'matchId', 'participants'])
    # these aren't needed? since we already have summoner ID

    info_temp = start_df.iloc[1, 1]
    #print('list of keys in info', info_temp.keys())
    # Now this is the complicated part
    # list of keys in meta dict_keys(['endOfGameResult', 'gameCreation', 'gameDuration', 'gameEndTimestamp', 'gameId', 'gameMode', 'gameName', 'gameStartTimestamp', 'gameType', 'gameVersion', 'mapId', 'participants', 'platformId', 'queueId', 'teams', 'tournamentCode'])

    # 'gameDuration' is interesting, and 'win' is my target

    start_df = start_df[['info', 'summoner_id']]
    flat_df = pd.json_normalize(start_df['info'].apply(flatten_nest_dict))  # apply the recursion
    print('Time:', (time.time() - start_time) / 60)
    print(flat_df.shape)
    start_df.reset_index(inplace=True, drop=True)
    flat_df.reset_index(inplace=True, drop=True)
    raw_df = pd.concat([start_df[['summoner_id']], flat_df], axis=1)  #
    # GAME MODE MUST BE CLASSIC:
    raw_df = raw_df[raw_df['gameMode'] == "CLASSIC"]

    # now we want to check those columns in the main data frame, and if the summonerid is in
    # there, keep the participant number in a new column
    id_columns = [col for col in flat_df.columns if any(substring in col for substring in [
        'summonerId'])]
    raw_df['participant_number'] = raw_df[['summoner_id'] + id_columns].apply(lambda row:
                                        find_participant_number(row, id_columns), axis=1)

    # now drop unneeded columns and unify column names:\
    reduce_cols_again = pd.DataFrame()
    team_cols = [i for i in raw_df.columns if 'team' in i.lower() and 'objective' in i.lower()] +\
                ['summoner_id']
    key_cols = key_col_holder()
    for i in range(1,11): #total count of participants
        temp = raw_df[raw_df['participant_number'] == str(i)]
        if temp.shape[0] == 0:
            continue # participant nubers are 9 or less!
        first_keep_partic_cols = [j for j in temp.columns if f'participants_{i}_' in j]  #OR
        #the 'teams' +'objectives' columns are interesting, but I only want the ones with the team
        # the  # summoner  # is on, using 'teamId'. Use apply to get the correct team id
        team_id_col = [i for i in first_keep_partic_cols if '_teamId' in i][0]
        temp['team_id_num'] = temp[team_id_col] # TODO: Loop and do this later? so that it
        # selects only the needed matching teams_ cols with interesting info
        # now rename columns:
        rename_dict = {k: k.replace(f'participants_{i}_', '') for k in first_keep_partic_cols}
        temp = temp.rename(columns = rename_dict)
        final_keep_partic_cols = team_cols + [j for j in rename_dict.values() if j in key_cols] # intentional redundency
        left_out_cols = [i for i in rename_dict.values() if i not in final_keep_partic_cols and i
        not in team_cols]
        print('length of left out cols', len(left_out_cols))
        temp = temp[final_keep_partic_cols]

        # do a redundent check that fails if the summoner IDs don't match
        bad_indices = temp[temp['summoner_id'] != temp['summonerId']].index
        if len(bad_indices) > 0:
            raise ValueError(f"index {bad_indices.tolist()} is mismatched")

        reduce_cols_again = pd.concat([reduce_cols_again, temp], axis=0).reset_index(drop=True)
        print('stacked', reduce_cols_again.shape)
    # now check columns again:
    print('Time:', (time.time() - start_time) / 60)
    #print(raw_df.columns)

    return raw_df


#Exploratory Analysis
def early_eda(raw_df, start_time):
    # How many nested columns are there?
    print(raw_df.shape) #(145,217, 3) less than a million rows, thats good!
    print(raw_df.head(5))


    start_df['matchId'] = start_df.metadata['matchId']
    start_df['gameDuration'] = start_df.metadata['matchId']




    # which ones seem interesting? check correlation

    #which ones are too obvious, and have to do with data leak?

    #which ones are most likely associated with early game decisions?


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
            # now batch read:
            df = flatten_and_reduce_df(df, start_time)
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
    #Now for EDA
    #early_eda(reduced_df, start_time)