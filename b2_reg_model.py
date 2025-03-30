from setup.setups import dir_base
import numpy as np
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

start_time = time.time()



#Linear Regression
def run_lin_reg(X_train, X_test, y_train, y_test):
    print('linear regression')
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    lin_reg_pred = lin_reg.predict(X_test)

    mse = mean_squared_error(y_test, lin_reg_pred)
    r2 = r2_score(y_test, lin_reg_pred)

    print('baseline linear regression model has MSE:', mse, 'and r squared:', r2)

    # now do a residuals plot:
    residuals = y_test - lin_reg_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(lin_reg_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()


#Random Forest Regression (Ensemble, different than XGBoosting)

def run_rf(X_train, X_test, y_train, y_test):
    rf_mod = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42) # just a
    # standard set up for now

    rf_mod.fit(X_train, y_train)
    rf_mod_pred = rf_mod.predict(X_test)

    mse = mean_squared_error(y_test, rf_mod_pred)
    mae = mean_absolute_error(y_test, rf_mod_pred)
    r2 = r2_score(y_test, rf_mod_pred)

    print('non tuned random forest regression performance. MSE:', mse, 'MAE:', mae, 'r-squared:',
          r2)

    # now do grid search typical tuning



if __name__ == '__main__':
    print('start!')
    past_run_date = '01-01-2025'

    X_train = pd.read_parquet(dir_base + f"data/reg_clean_data_{past_run_date}/x_train.parquet")
    X_test = pd.read_parquet(dir_base + f"data/reg_clean_data_{past_run_date}/x_test.parquet")
    y_train = pd.read_parquet(dir_base + f"data/reg_clean_data_{past_run_date}/y_train.parquet")
    y_test = pd.read_parquet(dir_base + f"data/reg_clean_data_{past_run_date}/y_test.parquet")

    X_train_reduc = pd.read_parquet(dir_base + f"data/reg_clean_data"
                                            f"_{past_run_date}/x_train_reduc.parquet")
    X_test_reduc = pd.read_parquet(dir_base + f"data/reg_clean_data"
                                            f"_{past_run_date}/x_test_reduc.parquet")
    print('final read in complete time:', (time.time() - start_time) / 60)