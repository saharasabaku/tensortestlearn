import os
from pathlib import Path
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout

import matplotlib
import matplotlib.pyplot as plt

from emnist import extract_training_samples, extract_test_samples

BASE_DIR = Path(__file__).resolve().parent
modeldata_dir = os.path.join(BASE_DIR,'textdata')

X_train, y_train = extract_training_samples('digits')
X_test, y_test = extract_test_samples('digits')

X_train, X_test = X_train/X_train.max(),X_test/X_test.max()

model = Sequential(
    [
        Flatten(input_shape=(28,28)),
        
        Dense(512),
        Activation('relu'),

        Dense(256),
        Activation('relu'),
        
        Dense(256),
        Activation('relu'),

        Dense(10),
        Activation('softmax')
    ]
)

#print(model.summary())

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    batch_size=50,
    epochs=30,
    verbose=1,
    validation_data=(X_test,y_test)
)

model.save(modeldata_dir)