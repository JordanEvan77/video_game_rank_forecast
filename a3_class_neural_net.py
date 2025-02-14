import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam





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

