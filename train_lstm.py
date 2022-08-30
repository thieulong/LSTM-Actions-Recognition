from typing import Sequence
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn import datasets
from sklearn.model_selection import train_test_split

punch_df = pd.read_csv("punch.txt")
neutral_df = pd.read_csv("neutral.txt")

X = []
y = []
no_of_timesteps = 20

datasets = punch_df.iloc[:,1:].values
n_samples = len(datasets)

for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i,:])
    y.append(1)

datasets = neutral_df.iloc[:,1:].values
n_samples = len(datasets)

for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i,:])
    y.append(0)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(optimizer="adam", metrics=["accuracy"],loss="binary_crossentropy")

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
model.save("lstm-model.h5")