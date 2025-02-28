from setup.setups import dir_base
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#plt.ion()

from video_game_rank_forecast.a0_reg_class_api_call import tiers_list
pd.options.mode.chained_assignment = None
#TODO: Make sure no time series factor is impactful
iter = 0
# I will kind of have two different datasets, for the different tasks
#Regression tasks: Overall tier/LP?
#Classification Tasks: Game to game victory, final tier placement

# lets start with the binary task of game to game victory.

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


def key_col_holder():
    key_columns = ['win', 'summonerId',
        "kills", # TODO: Too correlated? this is extremely obvious and correlated, still want to check on it though
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
        "assists", # TODO: Too correlated?
        "baronKills", # keep as this is if the player killed, not team
        "deaths", # TODO: Too correlated?
        "damageDealtToObjectives",
        "damageDealtToTurrets",
        "damageSelfMitigated",
        "dragonKills", # keep as this is if the player killed, not team
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

    #I want to remove attributes that are too closely tied to victory, and aren't able to be
    # influenced by the player. focusing on getting more kills is a lot like telling a race car
    # driver to just drive faster.
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
    # these aren't needed? since Ialready have summoner ID

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

    # now Iwant to check those columns in the main data frame, and if the summonerid is in
    # there, keep the participant number in a new column
    id_columns = [col for col in flat_df.columns if any(substring in col for substring in [
        'summonerId'])]
    raw_df['participant_number'] = raw_df[['summoner_id'] + id_columns].apply(lambda row:
                                        find_participant_number(row, id_columns), axis=1)

    # now drop unneeded columns and unify column names:\
    reduce_cols_again = pd.DataFrame()
    team_cols = [i for i in raw_df.columns if 'team' in i.lower() and 'objective' in i.lower()] +\
                ['summoner_id', 'team_id_num']
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

    #now do team divide and unify:
    team_list = ['100', '200'] # i think its this


    #team
    reduce_cols_again['team_id_num'] = reduce_cols_again['team_id_num'].astype('int64')
    team_0_df = reduce_cols_again[reduce_cols_again['team_id_num'] == 100]
    team_1_df = reduce_cols_again[reduce_cols_again['team_id_num'] == 200]
    team_cols_0 = [col for col in team_cols if 'teams_0' in col]
    team_cols_1 = [col for col in team_cols if 'teams_1' in col]

    # now drop all other teams columns and keep commons
    team_0_keep = [j for j in reduce_cols_again.columns if j not in team_cols_1]
    team_1_keep = [j for j in reduce_cols_again.columns if j not in team_cols_0]
    team_0_df = team_0_df[team_0_keep] # Iwant
    team_1_df = team_1_df[team_1_keep]

    #then a rename dictionary that unifies columes
    team_0_rename_dict = {k: k.replace(f'teams_0_', '') for k in team_cols_0}
    team_1_rename_dict = {k: k.replace(f'teams_1_', '') for k in team_cols_1}
    team_0_df = team_0_df.rename(columns=team_0_rename_dict)
    team_1_df = team_1_df.rename(columns=team_1_rename_dict)
    reduced_unified = pd.concat([team_0_df, team_1_df], axis=0)
    return reduced_unified


#Exploratory Analysis
def early_eda(raw_df, start_time):
    # How many nested columns are there?
    print(raw_df.shape) #(145,217, 3) less than a million rows, thats good!
    #print(raw_df.head(5))
    raw_df = raw_df.drop('objectives_horde', axis=1)
    bool_cols = raw_df.select_dtypes(include=['bool']).columns.tolist()

    # Select columns excluding boolean columns
    raw_df = raw_df.apply(lambda col: col.map(lambda x: pd.to_numeric(x, errors='ignore')))
    bool_cols = raw_df.select_dtypes(include=['bool']).columns
    raw_df[bool_cols] = raw_df[bool_cols].astype(int)
    cat_cols = raw_df.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    int_cols = raw_df.select_dtypes(include=['int']).columns.tolist()
    float_cols = raw_df.select_dtypes(include=['float']).columns.tolist()
    #summoner id is not a category, it is an identifier
    id_col = ['summoner_id']
    cat_cols.remove('summoner_id')
    cat_cols.remove('summonerId')


    # which ones seem interesting? check correlation
    corr_matrix = raw_df[float_cols + int_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=.5)
    plt.title('Correlation Heatmap of Numerical Columns')
    plt.savefig(dir_base + f"figures/correlation_heatmap.jpeg")
    #plt.show()

    #TODO:which ones are too obvious, and have to do with data leak?
    #which ones are most likely associated with early game decisions?


    #Look at distribution, what is skew in hist?
    raw_df.reset_index(inplace=True, drop=True)
    viz_num = int_cols + float_cols
    for i in viz_num:
        print('col:', i)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=raw_df, x=i, bins=10,
                     discrete=True)  # Adjust the number of bins as needed
        plt.title(f'Histogram of {i}')
        plt.xticks(rotation=45)
        plt.savefig(dir_base + f"figures/histograms/hist_dist_{i}.jpeg")
        #plt.show()
        print('')
        plt.close()
    #'AllnPings' are rare, with a heavy skew to left, mostly 0
    #Assist Pings are lesss rare, with a fair amount above 0, but mode is still 0
    #Assist kills are much closer to normal, but are still left skewed.
    #bait pings are again very left kewed, same as all in pings
    #Baron kills are surprisingly skewed to 0. This is a huge influencer of game victory,
    # and I would expect almost all games to have atleast 1 baron kill
    #Basic pings as a column looks useless?
    #bounty level is skewed left, but has an interesting slope to the right that isn't as steep
    # as I would have expected
    # histogram of deaths is closer to normal, which is also a bit surprising, I would have
    # expected right hand skew
    # Dragon kills are also surprisingly low, keep objectives instead!
    # inhibitors skewed left makes sense
    # horde kills are spiked at midle and edges, which is interesting! definitely needs scaling


    #all of this means that scaling will be necessary


    #TODO: Broken?
    # damageDealtToObjectives
    # damageDealtToTurrets
    # damageSelfMitigated




    #Look at outliers in box and whisker
    for j in viz_num:
        print('j', j)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=raw_df, x=j)
        plt.title(f'Box Plot of Values by {j}')
        plt.xlabel(j)
        plt.ylabel('Value')
        plt.savefig(dir_base + f"figures/box_plots//box_and_whisker_{j}.jpeg")
        #plt.show()
        plt.close()
    #as expected, most of these ahve heavy right side skew, that is dictated with outliers

    #CATEGORICAL EDA
    # for the different categories, what are the distributions?
    df_cat = pd.DataFrame(columns=['Column', 'Num Categories', 'Category Counts'])

    for col in cat_cols:
        num_categories = raw_df[col].nunique()
        category_counts = raw_df[col].value_counts().to_dict()
        null_count_pct = raw_df[col].isnull().sum() / len(raw_df) * 100
        new_row = pd.DataFrame({
            'Column': [col],
            'Num Categories': [num_categories],
            'Category Counts': [category_counts],
            'Null Count (%)': [null_count_pct]
        })
        df_cat = pd.concat([df_cat, new_row], ignore_index=True)

    df_cat.to_csv(dir_base + f"figures/cat_description.csv", index=False)
    # TODO: Horde first is so full of nulls, may be useless?

    #Items of import: ['objectives_baron_first', 'objectives_champion_first',
    # 'objectives_dragon_first', 'objectives_horde_first', 'objectives_inhibitor_first',
    # 'objectives_riftHerald_first', 'objectives_tower_first', 'firstBloodAssist', 'teamPosition',
    # 'win']



    #Stacked Categorical Counts Bar Graph with dependent variable coloring]
    for category in cat_cols:
        if category == 'win':
            continue #
        print('impact', category)
        pivot_df = raw_df.groupby([category]).agg({'win':'sum'})
        pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'Impact of {category} on win rate')
        plt.xlabel(category)
        plt.ylabel('Number of wins')
        plt.legend(title='Result')
        plt.savefig(dir_base + f"figures/bar_plots//impact_bar_plot_{j}.jpeg")
        #plt.show()
        plt.close()
    # any other key eda items to explore?
    # return nothing



def categorical_cleaning(cat_df, cat_cols):
    '''
    Would like to do both tasks worth of initial cleaning here, and then specifics later
    :param cat_df:
    :param cat_cols:
    :return:
    '''
    print('starting categorical cleaning')
    #check for duplicates across full dataframe

    # check for nulls:
    cat_nulls = pd.DataFrame(cat_df[cat_cols].isna().mean() * 100)
    cat_nulls.reset_index(inplace=True, drop=False)
    cat_nulls.columns = ['column', 'null_count']
    drop_cats = cat_nulls.loc[cat_nulls['null_count'] > 0.3, 'column'].unique()
    impute_cats = cat_nulls.loc[(cat_nulls['null_count'] <= 0.3) & (cat_nulls['null_count'] >
                                                                    0), 'column'].unique()
    # set up threshold:
    if len(drop_cats) > 0:
        cat_df = cat_df.drop(columns=drop_cats)

    #Categorical Cleaning and Encoding
    # which ones are ordinal?

    # Team position is  worth doing OHE
    cat_df = pd.get_dummies(cat_df, columns=['teamPosition'])
    cat_cols.remove('teamPosition')
    # would be good to label encode anything with more than say 10 categories.
    # Could do binary or hash encoding as an improvement
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le_cols = []
    for col in cat_cols:
        if len(cat_df[col].unique()) > 10:
            le_cols.append(col)
            cat_df[col] = le.fit_transform(cat_df[col])

    print('THese columns were Label Encoded', le_cols) #I am manually checking this

    #if there are a lot, try knn impute instead
    df_imputed = cat_df.copy()
    if len(impute_cats) > 0:
        print('imputing', impute_cats)
        impute = KNNImputer(n_neighbors=3)
        imputed_values = impute.fit_transform(cat_df[impute_cats])
        df_imputed[impute_cats] = imputed_values

    return df_imputed


def class_specific_cleaning(X, y):
    # class imbalance? Iwant to have an equal number of won and lost games
    # do Icare aobut rank, or should that be ignored?
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    return X_res, y_res




def drop_outliers(df, num_cols, threshold=1.5):
    '''
    OUtliers cleaner to count loss and decide if it should be floor cieling
    :param df:
    :param num_cols:
    :param threshold:
    :return:
    '''
    df_cleaned = df.copy()
    total_rows_lost = 0
    for col in num_cols:
        # IQR stuff
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)

        rows_before = df_cleaned.shape[0]
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

        rows_after = df_cleaned.shape[0]
        rows_lost = rows_before - rows_after
        total_rows_lost += rows_lost

        print(f'{col} outliers removed, dropped {rows_lost} rows')
    print(f'Total rows lost: {total_rows_lost}')

    return df_cleaned



def final_transforms_save_out(final_df, int_cols, float_cols):
    #now some transforms:
    #TODO: any features I should create through ratios or multiplication before impute and scale?
    temp_df  = final_df.copy()
    #final_df[int_cols] = final_df[int_cols].astype('int64') # shouldn't be needed
    num_cols = int_cols + float_cols

    # Numeric Cleaning
    num_nulls = pd.DataFrame(final_df[num_cols].isna().mean() * 100)
    num_nulls.reset_index(inplace=True, drop=False)
    num_nulls.columns = ['column', 'null_count']
    drop_nums = num_nulls.loc[num_nulls['null_count'] > 0.3, 'column'].unique().tolist()
    impute_nums = num_nulls.loc[(num_nulls['null_count'] <= 0.3) & (num_nulls['null_count'] > 0
                                                                    ), 'column'].unique().tolist()
    num_cols = [i for i in num_cols if i not in drop_nums and i!='win']
    # set up threshold:
    if len(drop_nums) > 0:
        print('dropping columns', final_df.shape)
        final_df = final_df.drop(columns=drop_nums)
        print('dropping columns', final_df.shape)
    # if there are a lot, try knn impute instead


    X_cols = [i for i in final_df.columns if i not in ['win', 'summoner_id', 'summonerid']]
    X = final_df[X_cols]
    y = final_df['win']

    #LDA using to reduce dimensionality for binary classification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    df_list = [X_train, X_test] # no need to impute y

    # impute
    for idx, num_df in enumerate(df_list):
        i = 0
        for col in impute_nums:
            i+=1
            print(i, col)
            median = num_df[col].median()
            num_df[col] = num_df[col].fillna(median)
        df_list[idx] = drop_outliers(num_df, num_cols, threshold=1.5) # TODO: The losses may be
        # too large here, may want to impute!

    # Do scaling
    X_train, X_test = df_list
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)
    X_test_standardized = scaler.transform(X_test)

    # now do balancing, within split groups
    X_train_standardized, y_train = class_specific_cleaning(X_train_standardized, y_train)
    X_test_standardized, y_test = class_specific_cleaning(X_test_standardized, y_test)

    # Dimension reduction
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_standardized, y_train)
    y_pred = lda.predict(X_test_standardized)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}") #90% accuracy is good for what I am doing~


    X_lda = lda.transform(X_train_standardized)
    plt.hist(X_lda[y_train == 0], alpha=0.5, label='Class 0')
    plt.hist(X_lda[y_train == 1], alpha=0.5, label='Class 1')
    plt.legend(loc='best')
    plt.title('LDA projection of the training data')
    plt.savefig(dir_base + f"figures/LDA_review.jpeg")
    # plt.show() # not too much overlap!

    X_train_lda = lda.transform(X_train_standardized)
    X_test_lda = lda.transform(X_test_standardized)
    print('lda result', X_train_lda.shape)

    return X_train_lda, X_test_lda, X_train_standardized, X_test_standardized, y_train, y_test
    # I want to compare both lda and non lda outputs



#TODO: When I have all of  this sorted out above, I should turn it into a COlumnTransformer that
# then gets fed to a pipeline, to make for really clean and efficient ETL.

#TODO: I should also run it on the full dataset instead of the half set!

def save_out_format(X_train_lda, X_test_lda, X_train, X_test, y_train, y_test, task='class'):
    os.makedirs(dir_base + f"data/{task}_clean_data_{past_run_date}", exist_ok=True)

    pd.DataFrame(X_train_lda).to_parquet(
        dir_base + f"data/{task}_clean_data_{past_run_date}/x_train_reduc.parquet")
    pd.DataFrame(X_test_lda).to_parquet(
        dir_base + f"data/{task}_clean_data_{past_run_date}/x_test_reduc.parquet")
    pd.DataFrame(X_train).to_parquet(
        dir_base + f"data/{task}_clean_data_{past_run_date}/x_train.parquet")
    pd.DataFrame(X_test).to_parquet(
        dir_base + f"data/{task}_clean_data_{past_run_date}/x_test.parquet")
    pd.DataFrame(y_train).to_parquet(
        dir_base + f"data/{task}_clean_data_{past_run_date}/y_train.parquet")
    pd.DataFrame(y_test).to_parquet(
        dir_base + f"data/{task}_clean_data_{past_run_date}/y_train.parquet")

    print('final save out complete time:', (time.time() - start_time) / 60)


if __name__ == '__main__':
    print('start!')
    start_time = time.time()
    past_run_date = '01-01-2025'

    parquet_high_name = dir_base + f"data/class_raw_data_{past_run_date}"

    common_columns = ['metadata', 'info', 'summoner_id']
    start_df = complex_read_in(parquet_high_name, tiers_list, common_columns)
    start_df.reset_index(inplace=True, drop=True)
    print('read in complete', (time.time() - start_time) / 60)
    #Now for EDA
    #early_eda(start_df, start_time)
    #Now that the result is smaller, do I want to batch clean? Or clean as one?

    # Select columns excluding boolean columns
    start_df = start_df.apply(
        lambda col: col.map(lambda x: int(x) if isinstance(x, float) and x.is_integer()
        else (float(x) if isinstance(x, float) else x)))
    #@TODO: Check that these work and then add to study guide
    bool_cols = start_df.select_dtypes(include=['bool']).columns.tolist()
    bool_cols.append('objectives_horde_first')
    start_df[bool_cols] = start_df[bool_cols].astype('Int64')
    int_cols = start_df.select_dtypes(include=['int']).columns.tolist()
    float_cols = start_df.select_dtypes(include=['float']).columns.tolist()
    cat_cols = [i for i in start_df.columns if i not in int_cols+float_cols]
    # summoner id is not a category, it is an identifier
    if len(int_cols) + len(float_cols) + len(cat_cols) != start_df.shape[1]:
        raise ValueError('column count doesnt align')
    id_col = ['summoner_id']
    cat_cols.remove('summoner_id')
    cat_cols.remove('summonerId')
    start_df = start_df.drop(columns=['summonerId'])
    cat_df = categorical_cleaning(start_df, cat_cols)
    X_train_lda, X_test_lda,X_train, X_test, y_train, y_test = final_df = \
        final_transforms_save_out(cat_df, int_cols, float_cols)
    save_out_format(X_train_lda, X_test_lda, X_train, X_test, y_train, y_test, task='class')


