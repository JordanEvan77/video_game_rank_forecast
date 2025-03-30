from setup.setups import dir_base
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

start_time = time.time()






def deep_learn_on_normal(X_train, X_test, y_train, y_test):
    #now take the full dataset and start on a shallow net, make deeper
    # once that works well do it on gan
    print('start net')

    #layers for a basic ffn
    input_layer = Input(shape=(X_train.shape[1],))
    hidden_layer_1 = Dense(units=64, activation='relu')(input_layer)
    hidden_layer_2 = Dense(units=32, activation='relu')(hidden_layer_1)
    output_layer = Dense(units=1)(hidden_layer_2)

    #model and compile
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='mse',
                  metrics=[MeanSquaredError(), RootMeanSquaredError()])

    # can adjust learning rate and learning schedule as well to help tune
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
    mse, rmse = model.evaluate(X_test, y_test, verbose=0)
    print('The MSE:', mse, 'and RMSE:', rmse)

    #Avoid over fitting here

    #TODO:once happy with the model, pickle it out


def set_up_gan():
    # to help create more synthetic data
    print('start gan')

    # return the synthetic data and run the below function again, and tune it as needed for the GAN
    # data


def deep_learn_on_gan(X_train, X_test, y_train, y_test, data_type='normal'):
    #now take the full dataset and start on a shallow net, make deeper
    # once that works well do it on gan
    print('start net')

    #layers for a basic ffn
    input_layer = Input(shape=(X_train.shape[1],))
    hidden_layer_1 = Dense(units=64, activation='relu')(input_layer)
    hidden_layer_2 = Dense(units=32, activation='relu')(hidden_layer_1)
    output_layer = Dense(units=1)(hidden_layer_2)

    #model and compile
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='mse',
                  metrics=[MeanSquaredError(), RootMeanSquaredError()])

    # can adjust learning rate and learning schedule as well to help tune
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
    mse, rmse = model.evaluate(X_test, y_test, verbose=0)
    print('The MSE:', mse, 'and RMSE:', rmse)

    #Avoid over fitting here

    #TODO:once happy with the model, pickle it out




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

    deep_learn_on_normal(X_train, X_test, y_train, y_test)

#TODO: I could always do a deep learning net, and create more synthetic data using a GAN if I
# wanted

# TODO: Once these are performing well, write up comparison of the model, and process to tune and
#  debug issues encountered.