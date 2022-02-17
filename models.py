import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class model1(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(256)
    def call(self, inputs):
        x = self.dense(inputs)
        #x = tf.nn.relu(x)
        x = tf.nn.softmax(x)
        return x


class model2(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(256)
        #self.relu = layers.ReLU()
        self.softmax = layers.Softmax()
    def call(self, inputs):
        x = self.dense(inputs)
        #x = self.relu(x)
        x - self.softmax(x)
        return x


a = model1()
a.build(input_shape=(None, 100))
a.summary()

b = model2()
b.build(input_shape=(None, 100))
b.summary()