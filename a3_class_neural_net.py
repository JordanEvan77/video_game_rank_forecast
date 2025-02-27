from setup.setups import dir_base
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam




def build_nn(X_train, X_test, y_train, y_test):

    #starting with a generic outline
    input_layer = Input(shape=(X_train.shape[1],))


    hidden_layer_1 = Dense(units=64, activation='relu')(input_layer)
    hidden_layer_2 = Dense(units=32, activation='relu')(hidden_layer_1)


    output_layer = Dense(units=1, activation='sigmoid')(hidden_layer_2)


    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    #start simple, can add lerning rate schedule, early stop, cross val, drop out, normalize, etc.
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    #binary classifier metrics: Accuracy, precision, recall, F1score, Log Loss
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)



if __name__ == '__main__':
    print('start!')
    past_run_date = '01-01-2025'
    X_train = pd.read_parquet(dir_base + f"data/clean_data_{past_run_date}/x_train.parquet")
    X_test = pd.read_parquet(dir_base + f"data/clean_data_{past_run_date}/x_train.parquet")
    y_train = pd.read_parquet(dir_base + f"data/clean_data_{past_run_date}/x_train.parquet")
    y_test = pd.read_parquet(dir_base + f"data/clean_data_{past_run_date}/x_train.parquet")