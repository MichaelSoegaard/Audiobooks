#Michael Søegaard 2019

import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import deque
import pickle
import time


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.config.experimental.list_physical_devices()

if not os.path.isdir("tuner"):
    os.makedirs("tuner")
if not os.path.isdir("logs"):
    os.makedirs("logs")

BATCH_SIZE = 64
EPOCHS = 5
NAME = "AudioB_V1"
LOG_DIR = f"tuner\\{int(time.time())}"

tensorboard = TensorBoard(log_dir=f"logs\\{NAME}")

#Load data
npz = np.load('data\\Audiobooks_data_train.npz')

X_train = npz['inputs'].astype(np.float)
y_train = npz['targets'].astype(np.int)

npz = np.load('data\\Audiobooks_data_test.npz')

X_test = npz['inputs'].astype(np.float)
y_test = npz['targets'].astype(np.int)



def create_model(hp):
    
    model = Sequential()
    
    model.add(Dense(hp.Int('input_units',
                                min_value=16,
                                max_value=160,
                                step=16), input_shape=(X_train.shape[1:])))
    model.add(Activation('relu'))
    
    for i in range(hp.Int('n_layers', 1, 2)):  # adding variation of layers.
        model.add(Dense(hp.Int(f'Dense_{i}_units',
                                min_value=32,
                                max_value=256,
                                step=32)))
        model.add(Activation('relu'))



    model.add(Dense(1, activation='sigmoid'))
       
       #Model compile settings:
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
           
       # Compile model
    model.compile(loss='binary_crossentropy',
               optimizer=opt,
               metrics=['accuracy']
               )

    return model


# In[14]:


tuner = RandomSearch(
    create_model,
    objective='val_accuracy',
    max_trials=100,  # how many variations on model?
    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
    directory=os.path.normpath('D:/'))

tuner.search_space_summary()

tuner.search(x=X_train,
             y=y_train,
             epochs=EPOCHS,
             batch_size=BATCH_SIZE,
             callbacks=[tensorboard],
             validation_data=(X_test, y_test))

tuner.results_summary()


with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)


history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard]
)