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

#startings out with a generic binary classifier nn


def build_nn(X_train, X_test, y_train, y_test):

    #starting with a generic outline for FFN
    input_layer = Input(shape=(X_train.shape[1],))

    hidden_layer_1 = Dense(units=64, activation='relu')(input_layer)
    hidden_layer_2 = Dense(units=32, activation='relu')(hidden_layer_1)

    output_layer = Dense(units=1, activation='sigmoid')(hidden_layer_2)


    model_bin = Model(inputs=input_layer, outputs=output_layer)
    model_bin.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    #start simple, can add lerning rate schedule, early stop, cross val, drop out, normalize, etc.
    #early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    # # also lower learning rate to help with performance getting stuck:
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
    model_bin.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    #binary classifier metrics: Accuracy, precision, recall, F1score, Log Loss
    loss, accuracy = model_bin.evaluate(X_test, y_test, verbose=0)
    print(loss, accuracy)
    print(classification_report(np.argmax(y_test, axis=1),
                                np.argmax(model_bin.predict([X_test]), axis=1)))



if __name__ == '__main__':
    print('start!')
    past_run_date = '01-01-2025'
    X_train = pd.read_parquet(dir_base + f"data/class_clean_data_{past_run_date}/x_train.parquet")
    X_test = pd.read_parquet(dir_base + f"data/class_clean_data_{past_run_date}/x_test.parquet")
    y_train = pd.read_parquet(dir_base + f"data/class_clean_data_{past_run_date}/y_train.parquet")
    y_test = pd.read_parquet(dir_base + f"data/class_clean_data_{past_run_date}/y_test.parquet")

    #Tensorflow adjustment
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    build_nn(X_train, X_test, y_train, y_test)

    X_train_reduc = pd.read_parquet(dir_base + f"data/class_clean_data"
                                            f"_{past_run_date}/x_train_reduc.parquet")
    X_test_reduc = pd.read_parquet(dir_base + f"data/class_clean_data"
                                            f"_{past_run_date}/x_test_reduc.parquet")



#TODO: Performance is too high, review baseline model