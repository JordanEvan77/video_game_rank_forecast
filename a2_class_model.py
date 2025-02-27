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
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

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
        'minchildweight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=37)
    gridsearch = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='rocauc', cv=cv,
                              n_jobs=-1, verbose=5)
    gridsearch.fit(X_train, y_train)
    bestparams = gridsearch.bestparams
    bestscore = gridsearch.bestscore
    print('best parameters', bestparams)

    #TODO: Reeview best params and make sure they make sense
    final_model = XGBClassifier(bestparams)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Final Accuracy: {accuracy}")
    print(f"Final Precision: {precision}")
    print(f"Final Recall: {recall}")
    print(f"Final F1 Score: {f1}")
    print(f"Final ROC-AUC: {roc_auc}")

    #Save out final model
    with open(dir_base + f"data/clean_data_{past_run_date}/x_train.parquet", 'wb') as file:
        pickle.dump(final_model, file)




if __name__ == '__main__':
    print('start!')
    past_run_date = '01-01-2025'
    X_train = pd.read_parquet(dir_base + f"data/clean_data_{past_run_date}/x_train.parquet")
    X_test = pd.read_parquet(dir_base + f"data/clean_data_{past_run_date}/x_train.parquet")
    y_train = pd.read_parquet(dir_base + f"data/clean_data_{past_run_date}/x_train.parquet")
    y_test = pd.read_parquet(dir_base + f"data/clean_data_{past_run_date}/x_train.parquet")