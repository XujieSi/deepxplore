# test shift

from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave

#from Model1 import Model1 as Model
#from Model2 import Model2 as Model#
#from Model3 import Model3 as Model


#from Model_nopool import ModelNoPool as Model
from Model_one_channel import ModelOneChannel as Model

import matplotlib.pyplot as plt
import numpy as np

from utils import shift_augmentation

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(_, _), (x_test, y_test) = mnist.load_data()

#x_test, y_test = shift_augmentation(x_test, y_test)

#print("x_test.shape:", x_test.shape)
#print("y_test.shape:", y_test.shape)


x_test, y_test = x_test[:400], y_test[:400]

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model = Model(input_tensor=input_tensor)


# scores = model.predict( x_test )
# ones = np.equal(np.argmax(scores, axis=1), y_test).sum()
# print("ones:", ones)
# exit()

def visualize( stats, figure=None):

    # To avoid creating figures per input sample, reuse the sample plot
    if figure is None:
        plt.ion()
        #plt.ioff()
        figure = plt.figure()
        figure.canvas.set_window_title('Visualization')
    else:
        figure.clf()


    K = len(stats)
    Row,Col = 1,K
    Row,Col = int((K+1)/2) ,2


    i = 0
    for image,scores,desc in stats:
        i += 1
        ax = figure.add_subplot(Row, Col, i)
        ax.axis('off')

        top_K = sorted(enumerate(scores), 
                key = lambda val : val[1], 
                reverse=True)
        display = [ "%d (%.6f%%)" % (x,y*100.0) for (x,y) in top_K ]

        #ax.set_ylabel(",\n".join(display[:3]) )
        #ax.set_ylabel( desc )
        ax.text(30, 25, desc + '\n' + '\n'.join(display[:3]) )

        # If the image is 2D, then we have 1 color channel
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        elif len(image.shape) == 3:
            ax.imshow(image)
        else:
            print("Error: unexpected image shape: ", image.shape)


        # Give the plot some time to update
        #plt.pause(0.01)

    plt.pause(0.5)
    plt.show()
    return figure

def save_stats(shows):
    print("shows size:", len(shows))
    figure = None
    i = 0
    while i < len(shows):
        e = min(i+10, len(shows) )
        figure = visualize( shows[i:e], figure)
        figure.savefig("stats_out/test_%d_%d.jpg" % (i,e))
        i += 10

def label_shift(i,j):
    s_text = "shift("
    if i != 0:
        s_text += ('U' if i < 0 else 'D') + '%d' % (np.absolute(i))
    else:
        s_text += 'N'

    s_text += ','
    if j != 0:
        s_text += ('L' if j < 0 else 'R') + '%d' % (np.absolute(j))
    else:
        s_text += 'N'
    s_text += ')'

    return s_text


def shift():
    ground_labels = y_test
    sample_X = x_test
    sample_Y = model.predict(sample_X)
    sample_labels = np.argmax(sample_Y, axis=1)

    print("sample_X.shape:", sample_X.shape)

    N = 1
    shows = []
    for i in range(-N,N+1):
        for j in range(-N,N+1):

            t1 = np.roll(sample_X, i, axis=1)
            ss = np.roll(t1, j, axis=2)

            scores = model.predict(ss)
            pred_labels = np.argmax(scores , axis=1)

            for k in range(len(ss)):
                if sample_labels[k] != ground_labels[k]:
                    #original prediction is wrong, skip
                    continue

                if pred_labels[k] == ground_labels[k]:
                    # shifted prediction is correct, skip
                    continue

                #if sample_Y[k][ sample_labels[k] ] < 0.9 or scores[k][ pred_labels[k] ] < 0.9: 
                    # if any has low confidence skip
                #    continue

                img0, img1 = sample_X[k].squeeze(), ss[k].squeeze()

                shows.append( (img0, sample_Y[k], "original(test%d)" % k) )
                shows.append( (img1, scores[k], label_shift(i,j) ) )
     
            print( "i=%d, j=%d" % (i,j), "shows size:", len(shows))


    #fig = None
    #fig = visualize(shows[:10], fig)
    save_stats(shows)

shift()
