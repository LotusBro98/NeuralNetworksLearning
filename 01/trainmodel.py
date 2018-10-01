import gc
import pickle
import time

import numpy as np
import PIL.Image
import itertools
from IPython.core.display import Image, display
import scipy.ndimage
import random
import numpy as np
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
import tqdm
import math
#%matplotlib inline

def processNeurons(features, weights):
    return (1. / (1. + np.exp(-np.dot(features, weights))))




features = [[0,1], [1,2], [2,4], [3,4], [4,4], [5,4], [6,8], [7,9],
            [1,5], [3,8], [4, 12], [7, 15]]
features = np.vstack(features)
likelihoodsTrue = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
Weights = [0.01, 0.01]

def loss(w):
    return np.sum(np.abs(np.subtract(processNeurons(features, w), likelihoodsTrue)))


rx=16
xx1, xx2 = np.meshgrid(np.linspace(-rx, rx, 50), np.linspace(-rx, rx, 50))
xx = np.vstack([xx1.ravel(), xx2.ravel()]).T
def draw_model(w):
    p = (processNeurons(xx, w)).reshape(xx1.shape)
    return plt.imshow(p, cmap=plt.cm.PuOr_r, extent=(xx1.min(), xx1.max(), xx2.min(), xx2.max()), origin='lower')



def draw_points(features):
    f = features
    return plt.scatter(f[:, 0], f[:, 1], c=likelihoodsTrue, cmap=plt.cm.Set1, edgecolor='k')


def epoch(args):
    #Weights = [np.cos(args * -0.01), np.sin(args* -0.01)]

    L = loss(Weights)
    #print(L, Weights)
    dw = 0.05
    speed = 0.01
    for i in range(len(Weights)):
        Weights[i] = Weights[i] + dw
        dL = loss(Weights) - L
        Weights[i] = Weights[i] - dw - dL * speed

    if (args % 10 == 0):
        draw_model(Weights)
        draw_points(features)
        print(L)

    gc.collect()


gc.enable()

anim = FuncAnimation(plt.figure(), epoch, frames=500, interval=1)
plt.show()



# with open('./datasets/hw_1_train.pickle', 'rb') as f:
#     train = pickle.load(f)
#
# plt.imshow(train['data'][0].reshape(28,28))
# plt.show()
