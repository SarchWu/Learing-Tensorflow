# -*- coding: utf-8 -*-
# @Time    : 2019/5/29 14:49
# @Author  : wuqi
# @Site    : 
# @File    : tf_CNN.py.py
# @Software: PyCharm
import tensorflow as tf
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        #mnist gray 28*28
        #padding =same means the output feature map size is the same with the input feature map
        self.conv1=tf.keras.layers.Conv2D(filters=32,kernel_size=(5,5),padding="same",activation=tf.nn.relu)
        self.pooling1=tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2)
        self.conv2=tf.keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding="same",activation=tf.nn.relu)
        self.pooling2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        #tf.keras.layer.Flatten(),remains the number axis=0(batch_size) dimension
        self.flatten=tf.keras.layers.Flatten()
        self.dense1=tf.keras.layers.Dense(units=1024)
        self.dense2 = tf.keras.layers.Dense(units=64)
        self.dense3=tf.keras.layers.Dense(units=10)
    def __call__(self, inputs):
        input=tf.reshape(inputs,[-1,28,28,1])
        print ("inputs shape ：",input.shape)

        x=self.conv1(input)
        x=self.pooling1(x)
        x=self.conv2(x)
        x =self.pooling2(x)
        x =self.flatten(x)
        x=self.dense1(x)
        x=self.dense2(x)
        x=self.dense3(x)
        return x

    def predict(self,inputs):
        print ("start predicting")
        logits=self(inputs)
        print ("counting logits down")
        # tf.argmax returns the index of the max number
        print (" logits shape: ",logits.shape)
        return tf.argmax(logits,axis=-1)

