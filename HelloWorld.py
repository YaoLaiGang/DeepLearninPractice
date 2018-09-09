# -*- coding: UTF-8 -*-
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD,Adam
from keras.activations import sigmoid , relu

#维度转化，需要reshape narray的维度，令二维转化为一维，并使用to_categorical将结果向量化
(x_train,y_train) , (x_test,y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape)[0],28*28)
x_train = x_train.astype('float32')#如果不转化，会变成很多0的0矩阵
x_train = x_train / 255 #图像多使用灰阶区0-1,不使用准确率可能会下降
x_test = x_test.reshape(x_test.shape[0],28*28)
x_test = x_test.astype('float32')
x_test = x_test / 255
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)
#print(x_train[0])
#exit()

#三个sigmoid 一个softmax
model = Sequential()
model.add(Dense(input_dim = 28*28,units=500,activation='relu'))
model.add(Dropout(0.5))#当拟合过度而test太低时，使用dropout
model.add(Dense(units=500,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=500,activation='relu'))
model.add(Dropout(0.5))

#for i in range(10) : #增加多层以期望得到更好的效果，但是sigmoid会有训练不足卡住的反作用,此时我们可以换成RELU
#    model.add(Dense(units=500,activation='relu'))

#mse 均方误差，常用于连续变量，不适用于离散变量，分类多用categorical_crossentropy
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='categorical_crossentropy',
optimizer='adam', #adam 上升速度更快，比ＳＧＢ好很多 , lenrning rate 选取方法
metrics=['accuracy']
)

#batch_size 太大，如果使用GPU速度就会变快，但是准确率会降低许多
#batch_size 太小，无法发挥GPU的优势
model.fit(x_train,y_train,batch_size=100,epochs=20)

scores = model.evaluate(x_train,y_train,batch_size=10000) #观察拟合的实际情况
resule = model.evaluate(x_test,y_test,batch_size=10000)

print('Test Acc : ',resule[1])#第一个是错误率 cross entropy/mse
print('Train Acc :' ,scores[1])#第二个是准确率

