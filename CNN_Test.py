# -*- coding: UTF-8 -*-
#使用卷积神经网络实现手写数字识别
from keras.layers.core import Dense , Flatten , Activation
from keras.activations import relu , softmax
from keras.losses import categorical_crossentropy
from keras.layers import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.datasets import mnist
from keras.optimizers import adam
from keras.utils.np_utils import to_categorical
from keras.models import Sequential

#从 mnist中获取数据
(x_train,y_train) , (x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1) # 行数不定，列数为1的28*28的图片，如果是RGB列数为3
x_test = x_test.reshape(-1,28,28,1)

#将结果向量化，以方便处理
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

#将图片灰度化，以更容易计算
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

#模型初始化
model = Sequential()
#1*28*28
model.add(Conv2D(
    25,
    (3,3), #指定滤波器的个数，大小
    input_shape = (28,28,1),
    border_mode = 'same'
))
#25*26*26
model.add(MaxPool2D(2,2)) #多少维度矩阵取最大值
#25*13*13
model.add(Conv2D(50,(3,3)))
#50*11*11
model.add(MaxPool2D(2,2))
#50*5*5

#拉成线状数据 使用Flatten
model.add(Flatten())

#放入full connect network
model.add(Dense(units=500,activation = 'relu'))

model.add(Dense(units = 10,activation = 'softmax'))

#编译
model.compile(optimizer='adam',
loss=categorical_crossentropy,
metrics=['accuracy']
)

model.fit(x_train,y_train,batch_size=32,epochs=20) 

result = model.evaluate(x_train,y_train,batch_size=32)
acc = model.evaluate(x_test,y_test,batch_size=32)

print('tarin acc : ',result[1])
print('test acc : ',acc[1] )