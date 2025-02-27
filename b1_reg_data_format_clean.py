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
from video_game_rank_forecast.a1_class_data_format_clean import key_col_holder
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


def agg_for_task(clean_df):
    # used to bring up to level necessary for overall placement
    print('aggregating for regression')

final_rank = '' # will need to read in from original dataset and use funciton converter
