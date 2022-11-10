import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import random
sources = pd.read_csv('sources_united.csv')
l={'Conventional Demands':[]}
x = pd.DataFrame(data=l)
x["Conventional Demands"] =sources["Coal"].fillna(0)+sources["Large Hydro merged"].fillna(0)+sources["Natural Gas merged"].fillna(0)
x = x.values.tolist()
train_data=x[:80]
test_data=x[80:]


def splitSequence(seq, n_steps):
    X = []
    y = []

    for i in range(len(seq)):
        lastIndex = i + n_steps

        if lastIndex > len(seq) - 1:
            break

        seq_X, seq_y = seq[i:lastIndex], seq[lastIndex]

        X.append(seq_X)
        y.append(seq_y)
        pass

    X = np.array(X)
    y = np.array(y)

    return X, y


n_steps = 5
X, y = splitSequence(train_data, n_steps = 5)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
model = tf.keras.Sequential()
model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
model.fit(X, y, epochs=3000, verbose=1)
test_X, test_y = splitSequence(test_data, n_steps=5)
c=random.randrange(1,len(test_y))
test_data = test_X[c]
test_data = test_data.reshape((1, n_steps, n_features))
predictNextNumber = model.predict(test_data, verbose=1)
print("Program predicted:")
print(predictNextNumber)
print("Actual Value was:")
print(test_y[c])