# train model


from __future__ import print_function


from keras.datasets import mnist
from keras.layers import Input

#from Model1 import Model1 as Model
#from Model2 import Model2 as Model
#from Model3 import Model3 as Model
#from Model_nopool import ModelNoPool as Model

#from Model_gd import Model_gd as Model
#from Model_one_channel import ModelOneChannel as Model
from Model_capbase import Model_capbase as Model

import numpy as np


# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model = Model(input_tensor=input_tensor,train=True)
#model = Model(input_tensor=input_tensor)


# the data, shuffled and split between train and test sets
# (_, _), (x_test, y_test) = mnist.load_data()
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# x_test = x_test.astype('float32')
# x_test /= 255

# scores = model.predict( x_test )
# ones = np.equal(np.argmax(scores, axis=1), y_test).sum()
# print("ones:", ones)
