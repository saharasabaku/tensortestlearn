import os
from pathlib import Path
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import models
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


def predict():
    #model = Sequential([
    #    Flatten(input_shape=(28,28)),
    #    
    #    Dense(512),
    #    Activation('relu'),
    #    Dense(256),
    #    Activation('relu'),
    #    
    #    Dense(256),
    #    Activation('relu'),
    #    Dense(10),
    #    Activation('softmax')
    #])
#
    #model.compile(
    #    optimizer='adam',
    #    loss='sparse_categorical_crossentropy',
    #    metrics=['accuracy']
    #)

    model = models.load_model(modeldata_dir)
    return model



def main(i):
    i = i
    X     = []
    data  = X_test[i]
    X.append(data)
    X     = np.array(X)
    
    # モデル呼び出し
    model = predict()
    
    # numpy形式のデータXを与えて予測値を得る
    model_output = model.predict([X])[0]
    # 推定値 argmax()を指定しmodel_outputの配列にある推定値が一番高いインデックスを渡す
    predicted = model_output.argmax()
    # アウトプット正答率
    accuracy = int(model_output[predicted] *100)
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"{predicted}  {y_test[i]}")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


for i in range(5):
    main(i)