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
EPOCHS = 30
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



def create_model():
    
    model = Sequential()
    
    model.add(Dense(160, input_shape=(X_train.shape[1:]), activation='relu'))
    
    model.add(Dense(96, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
       
       #Model compile settings:
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
           
       # Compile model
    model.compile(loss='binary_crossentropy',
               optimizer=opt,
               metrics=['accuracy']
               )

    return model

model = create_model()
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test)
)