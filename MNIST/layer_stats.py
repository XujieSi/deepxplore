

from __future__ import print_function

import argparse

import keras as K
from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave

#from Model1 import Model1 as Model
#from Model2 import Model2 as Model
#from Model3 import Model3 as Model

from Model_nopool import ModelNoPool as Model


import matplotlib.pyplot as plt
import numpy as np



# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(_, _), (x_test, y_test) = mnist.load_data()


#idx = 73
idx=78

x_test, y_test = x_test[idx:idx+1], y_test[idx:idx+1]

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model = Model(input_tensor=input_tensor)


tmp = np.roll(x_test, -1, axis=1)
x_U1L1 = np.roll(tmp, -1, axis=2)

scores1 = model.predict(x_test)
print("scores1:", scores1)
print("predict:", np.argmax(scores1, axis=1))

scores2 = model.predict(x_U1L1)
print("scores2:", scores2)
print("predict:", np.argmax(scores2, axis=1))

#scores2 = model.get_layer('before_softmax').output
#print("scores2:", scores2)

layer_name='before_softmax'
layer_name='block1_conv1'
layer_name='block3_conv1'
#layer_name="fc1"
intermediate_layer_model = K.models.Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

intermediate_output1 = intermediate_layer_model.predict(x_test)
a = intermediate_output1

# a = np.transpose(a.squeeze(), (2,0,1))
# print("a.shape:", a.shape)
# print("inter1: \n", a[0][0:3])

intermediate_output2 = intermediate_layer_model.predict(x_U1L1)
b = intermediate_output2

# b = np.transpose(b.squeeze(), (2,0,1))
# print("b.shape:", b.shape)
# print("inter2: \n", b[0][0:3])

sz = 10
ct = 0
for i in range(sz-1):
	for j in range(sz-1):
		a = intermediate_output1[0][i+1][j+1]
		b = intermediate_output2[0[i][j]

		if  np.absolute((a-b).sum()) < 0.001:
			ct += 1
			print("no difference: (%d,%d)" % (i,j))

print("ct=", ct)
