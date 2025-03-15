from setup.setups import dir_base
import numpy as np
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pickle
start_time = time.time()



#Linear Regression
def run_lin_reg(X_train, X_test, y_train, y_test):
    print('linear regression')









#Random Forest Regression (Ensemble, different than XGBoosting)

def run_rf(X_train, X_test, y_train, y_test):
    rf_mod = RandomForestRegressor()


# USE CURRENT DATASET TO RUN BASE MODELS