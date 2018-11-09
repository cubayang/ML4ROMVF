# grid search hyperparameters with scikit-learn
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras import initializers
from keras import optimizers
from sklearn.model_selection import KFold, GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to create model, required for KerasRegressor
def create_model(init_mode='random_normal'):
    # create model
    model = Sequential()
    # input layer
    model.add(Dense(256, kernel_initializer=init_mode, activation='relu', input_shape=(train_data.shape[1],)))
    # hidden layer
    model.add(Dense(64,  kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(16,  kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(4,   kernel_initializer=init_mode, activation='relu'))
    # output layer
    model.add(Dense(1,   kernel_initializer=init_mode, activation='linear'))
    # compile model
    model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['mae'])
    return model

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
# create model
model = KerasRegressor(build_fn=create_model, epochs=10000, batch_size=128, verbose=0)
# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'random_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(train_data, train_targets)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
