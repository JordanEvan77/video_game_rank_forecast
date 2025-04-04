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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from video_game_rank_forecast.a0_reg_class_api_call import tiers_list
from video_game_rank_forecast.a1_class_data_format_clean import key_col_holder, complex_read_in, \
    flatten_nest_dict,find_participant_number, flatten_and_reduce_df, categorical_cleaning, \
    drop_outliers,save_out_format

pd.options.mode.chained_assignment = None

# This will again take some reshaping. Iprobably want some or average of the key variables?
# Or something else that shows game to game improvement? Iwant to measure how someone can climb
# and improve, so maybe difference between the min and max, to show improvment? Or difference
# between average early on and average later on?


#I will need to convert the tier, rank and LP into a single integer, like a sliding scale,
# and merge that in from the all tier summoner ID lists I created. Idon't have their initial rank,
# just their final rank. would be interesting to take wins and losses to get games played,
# and use that as an attribute too.


def convert_rank_to_int():
    '''
    A way to turn this into an regression task
    :param tier:
    :param rank:
    :param lp:
    :return:
    '''

    rank_df = pd.read_csv(dir_base + 'data\\all_tier_summoner_ids.csv')
    tier_dict = {'IRON': 1000, 'BRONZE': 2000, 'SILVER': 3000, 'GOLD': 4000, 'PLATINUM': 5000,
        'DIAMOND': 6000}
    rank_dict = {'IV': 100, 'III': 200, 'II': 300, 'I': 400}
    rank_df['final_rank'] = rank_df.apply(lambda row: tier_dict[row['tier']] +
                                                      rank_dict[row['rank']] +
                                                      int(row['LP']), axis=1)
    # this is now a continuous target

    return rank_df[['summoner_id', 'final_rank']].sort_values(
        by='final_rank', ascending=False).drop_duplicates(keep='first')#final_value


def agg_for_reg_task(final_df, int_cols, float_cols): # this is different than
    # final_transforms_save_out(final_df,
    # int_cols, float_cols):
    # used to bring up to level necessary for overall placement

    num_cols = int_cols + float_cols
    # Numeric Cleaning
    num_nulls = pd.DataFrame(final_df[num_cols].isna().mean() * 100)
    num_nulls.reset_index(inplace=True, drop=False)
    num_nulls.columns = ['column', 'null_count']
    drop_nums = num_nulls.loc[num_nulls['null_count'] > 0.3, 'column'].unique().tolist()
    impute_nums = num_nulls.loc[(num_nulls['null_count'] <= 0.3) & (num_nulls['null_count'] > 0
                                                                    ), 'column'].unique().tolist()
    num_cols = [i for i in num_cols if i not in drop_nums and i != 'win']
    # set up threshold:
    if len(drop_nums) > 0:
        print('dropping columns', final_df.shape)
        final_df = final_df.drop(columns=drop_nums)
        print('dropping columns', final_df.shape)

    #Get max rank:
    rank_df = convert_rank_to_int() #

    #merge
    merge_df = pd.merge(final_df, rank_df, how='inner', on='summoner_id') # will drop folks
    # without a final rank
    print('goes from :', final_df.shape[0], merge_df.shape[0])

    # Now do agg
    print('aggregating for regression')
    #Reset Index to get ID back
    final_grp = merge_df.groupby(['summoner_id', 'final_rank']).mean().reset_index(drop=False)

    # are there a couple that would be interesting to get the max from? To show off their most
    # impressive game?


    X_cols = [i for i in final_grp.columns if i not in ['final_rank', 'summoner_id', 'summonerid']]
    X = final_grp[X_cols]
    y = final_grp['final_rank']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    df_list = [X_train, X_test]  # no need to impute y

    # impute?
    for idx, num_df in enumerate(df_list):
        i = 0
        for col in impute_nums:
            i += 1
            print(i, col)
            median = num_df[col].median()
            num_df[col] = num_df[col].fillna(median)
        df_list[idx] = drop_outliers(num_df, num_cols, threshold=1.5)
    #TODO: lots of rows lost, will need to review


    # Do scaling
    X_train, X_test = df_list
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)
    X_test_standardized = scaler.transform(X_test)


    #Dimensionality reduction?
    pca = PCA()
    pca.fit(X_train_standardized)

    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_explained_variance >= 0.90) + 1
    print('90% variance is captured at', num_components)

    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_.cumsum(), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

    #pick number of componnets if the graph looks right
    pca_final = PCA(n_components=num_components)
    pca_final.fit(X_train_standardized) # only 3 PCs needed

    X_train_pca = pca_final.transform(X_train_standardized)
    X_test_pca = pca_final.transform(X_test_standardized)

    return X_train_pca, X_test_pca, X_train_standardized, X_test_standardized, y_train, y_test





if __name__ == '__main__':
    print('start!')
    start_time = time.time()
    past_run_date = '01-01-2025'

    parquet_high_name = dir_base + f"data/class_raw_data_{past_run_date}"

    common_columns = ['metadata', 'info', 'summoner_id'] # need to find rank and tier in this
    start_df = complex_read_in(parquet_high_name, tiers_list, common_columns, start_time)
    start_df.reset_index(inplace=True, drop=True)
    print('read in complete', (time.time() - start_time) / 60)

    # Select columns excluding boolean columns
    start_df = start_df.apply(
        lambda col: col.map(lambda x: int(x) if isinstance(x, float) and x.is_integer()
        else (float(x) if isinstance(x, float) else x)))
    bool_cols = start_df.select_dtypes(include=['bool']).columns.tolist()
    bool_cols.append('objectives_horde_first')
    start_df[bool_cols] = start_df[bool_cols].astype('Int64')
    int_cols = start_df.select_dtypes(include=['int']).columns.tolist()
    float_cols = start_df.select_dtypes(include=['float']).columns.tolist()
    cat_cols = [i for i in start_df.columns if i not in int_cols+float_cols]
    # summoner id is not a category, it is an identifier
    if len(int_cols) + len(float_cols) + len(cat_cols) != start_df.shape[1]:
        raise ValueError('column count doesnt align')
    # id_col = ['summoner_id']
    cat_cols.remove('summoner_id')
    cat_cols.remove('summonerId')
    start_df = start_df.drop(columns=['summonerId'])


    cat_df = categorical_cleaning(start_df, cat_cols)


    X_train_pca, X_test_pca, X_train, X_test, y_train, y_test = agg_for_reg_task(cat_df,
                                                                                 int_cols, float_cols)

    save_out_format(start_time, X_train_pca, X_test_pca, X_train, X_test, y_train, y_test,
                    task='reg', past_run_date=past_run_date)


# completed, ready for larger run on full dataset

