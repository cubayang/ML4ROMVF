# k-fold cross validation
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras import initializers
from keras import optimizers
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load training dataset
train_data = pd.read_csv("train_data.csv", usecols=['position', 'area', 'diameter', 'angleup', 'angledown', 'shape', 'pdrop', 'reloc'])
train_targets = pd.read_csv("train_targets.csv", usecols=['Fr'])
# feature scaling and mean normalization for train data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
train_data = train_data.values
train_targets = train_targets.values.flatten('F')
# define k-fold cross validation test harness
kfold = KFold(n_splits=10, shuffle=False, random_state=seed)
cvscores = []
for train, test in kfold.split(train_data, train_targets):
    # create model
    model = Sequential()
    # input layer
    model.add(Dense(256, kernel_initializer='random_normal', activation='relu', input_shape=(train_data.shape[1],)))
    # hidden layer
    model.add(Dense(64,  kernel_initializer='random_normal', activation='relu'))
    model.add(Dense(16,  kernel_initializer='random_normal', activation='relu'))
    model.add(Dense(4,   kernel_initializer='random_normal', activation='relu'))
    # output layer
    model.add(Dense(1,   kernel_initializer='random_normal', activation='linear'))
    # compile model
    model.compile(optimizer='Nadam', loss='mean_squared_error', metrics=['mae'])
    # Fit the model
    model.fit(train_data[train], train_targets[train], epochs=10000, batch_size=128, verbose=0)
    # evaluate the model
    scores = model.evaluate(train_data[test], train_targets[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
