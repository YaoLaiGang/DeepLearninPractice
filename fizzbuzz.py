# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.activations import relu , softmax
from keras.optimizers import adam
from keras.layers.core import Dense , Activation

#经典的编码转化函数
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

NUM_DIGITS = 10
x_train = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)]) #二进制101-1024
y_train = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])#对应的fizz buzz类

model = Sequential()
model.add(Dense(input_dim = 10,units = 5000,activation = "relu"))
model.add(Dense(units = 4,activation = "softmax"))

model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)

model.fit(x_train,y_train,batch_size=100,epochs=100)

result = model.evaluate(x_train,y_train,batch_size=1000)

print(result[1])
