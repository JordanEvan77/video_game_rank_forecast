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


def convert_rank_to_int(tier, rank, lp):
    '''
    A way to turn this into an regression task
    :param tier:
    :param rank:
    :param lp:
    :return:
    '''
    tier_values = {'Iron': 1000, 'Bronze': 2000, 'Silver': 3000, 'Gold': 4000, 'Platinum': 5000,
        'Diamond': 6000}
    rank_values = {'IV': 100, 'III': 200, 'II': 300, 'I': 400 #TODO: isn't there a rank 5?
    }
    tier_value = tier_values[tier]
    rank_value = rank_values[rank]
    final_value = tier_value + rank_value + lp

    return final_value


def agg_for_reg_task(clean_df, int_cols, float_cols): # this is different than
    # final_transforms_save_out(final_df,
    # int_cols, float_cols):
    # used to bring up to level necessary for overall placement
    print('aggregating for regression')

final_rank = '' # will need to read in from original dataset and use funciton converter





if __name__ == '__main__':
    print('start!')
    start_time = time.time()
    past_run_date = '01-01-2025'

    parquet_high_name = dir_base + f"data/class_raw_data_{past_run_date}"

    common_columns = ['metadata', 'info', 'summoner_id'] # need to find rank and tier in this
    start_df = complex_read_in(parquet_high_name, tiers_list, common_columns)
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
    id_col = ['summoner_id']
    cat_cols.remove('summoner_id')
    cat_cols.remove('summonerId')
    start_df = start_df.drop(columns=['summonerId'])
    cat_df = categorical_cleaning(start_df, cat_cols)
    X_train, X_test, y_train, y_test = final_df = agg_for_reg_task(cat_df, int_cols, float_cols)
    save_out_format(X_train, X_test, y_train, y_test)