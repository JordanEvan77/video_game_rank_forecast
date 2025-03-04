from setup.setups import dir_base
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,\
    classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pickle
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

start_time = time.time()

def log_model(X_train, X_test, y_train, y_test):
    #Logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(class_report) # just as a baseline for the xgboost



def xgb_model(X_train, X_test, y_train, y_test):
    #XGBOOST
    xgb = XGBClassifier(eval_metric='logloss')

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_pred_proba = xgb.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC-AUC: {roc_auc}")

    #TUNE with cross validation and regularization.
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=37)
    gridsearch = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='roc_auc', cv=cv,
                              n_jobs=-1, verbose=1)
    gridsearch.fit(X_train, y_train)
    bestparams = gridsearch.best_params_
    bestscore = gridsearch.best_params_
    print('grid search time:', (time.time()-start_time)/60)
    print('best parameters', bestparams)

    #TODO: Reeview best params and make sure they make sense
    final_model = XGBClassifier(colsample_bytree=bestparams['colsample_bytree'],
                                gamma=bestparams['gamma'],
                                learning_rate=bestparams['learning_rate'],
                                max_depth=bestparams['max_depth'],
                                n_estimators=bestparams['n_estimators'],
                                subsample=bestparams['subsample'])
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Best Parameters Final Accuracy: {accuracy}")
    print(f"Best Parameters Final Precision: {precision}")
    print(f"Best Parameters Final Recall: {recall}")
    print(f"Best Parameters Final F1 Score: {f1}")
    print(f"Best Parameters Final ROC-AUC: {roc_auc}")
    return final_model



#TODO: Add explainable portion of analytics, why is it so accurate, what is the driver?
def xbg_feat_import(final_model):

    #weight and gain are most useful for my purposes
    weight_importance = final_model.get_booster().get_score(
        importance_type='weight')  # You can use 'weight', 'gain', or 'cover'
    weight_importance_df = pd.DataFrame(weight_importance.items(), columns=['Feature',
                                                                            'Importance'])

    gain_importance = final_model.get_booster().get_score(
        importance_type='weight')  # You can use 'weight', 'gain', or 'cover'
    gain_importance_df = pd.DataFrame(gain_importance.items(), columns=['Feature',
                                                                            'Importance'])

    plot_importance(final_model, importance_type='weight')  # You can use 'weight', 'gain', or 'cover'
    plt.show()

    #Save out final model
    with open(dir_base + f"data/clean_data_{past_run_date}/xgboost_best_model.pkl", 'wb') as file:
        pickle.dump(final_model, file)
    print('Done', (time.time()-start_time)/60)
    #TODO: Look at the stuff above and save it out, plus any key take aways, write out here


if __name__ == '__main__':
    print('start!')
    past_run_date = '01-01-2025'

    X_train = pd.read_parquet(dir_base + f"data/class_clean_data_{past_run_date}/x_train.parquet")
    X_test = pd.read_parquet(dir_base + f"data/class_clean_data_{past_run_date}/x_test.parquet")
    y_train = pd.read_parquet(dir_base + f"data/class_clean_data_{past_run_date}/y_train.parquet")
    y_test = pd.read_parquet(dir_base + f"data/class_clean_data_{past_run_date}/y_test.parquet")

    X_train_reduc = pd.read_parquet(dir_base + f"data/class_clean_data"
                                            f"_{past_run_date}/x_train_reduc.parquet")
    X_test_reduc = pd.read_parquet(dir_base + f"data/class_clean_data"
                                            f"_{past_run_date}/x_test_reduc.parquet")
    print('final read in complete time:', (time.time() - start_time) / 60)

    log_model(X_train, X_test, y_train, y_test) # y isn't the rignt size
    xgb_model(X_train, X_test, y_train, y_test)

    #now with reduced dimension:
    log_model(X_train_reduc, X_test_reduc, y_train, y_test)
    final_model = xgb_model(X_train_reduc, X_test_reduc, y_train, y_test)
    xbg_feat_import(final_model)


#TODO: it is important to find out which items are contributing to the models high accuracy,
# and determine if the items are data leak of sorts, and should be removed. Since this is a
# coaching mechanism, there are things teh player can't control .


