'''
Bennett Garza
10/19/2018
Machine Learning Astrophotography Post-Processing (MLAPP)

Image_Network.py
A machine learning network to train on post-processing styles
'''

import numpy as np
import math

a = 0.01

class Image_Network(object):

    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes]
        self.weights = [np.random.randn(y, x)
                for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    # For inner layers
    def leaky_ReLU(self, x):
        if x < 0:
            return a*x
        else:
            return x

    def leaky_ReLU_prime(self, x):
        if x < 0:
            return a
        else:
            return 1

    # For output layer
    def tanh(self, x):
        return math.tanh(x)

    def tanh_prime(self, x):
        return 1-math.tanh(x)*math.tanh(x)

    # Transforms 0 to 255 range of pixel to -1 to 1
    def normalize(self, img):
        return img/128 - 1

    # Transforms -1 to 1 output range to 0 to 255 pixel range
    def denormalize(self, out):
        return 128*(out+1)

    # Returns output of network given normalized input
    # Uses leaky ReLU for all but output layers, and tanh for output layer
    def feed_forward(self, nA):
        for i in range(1, self.num_layers-1):
            nA = leaky_ReLU(np.dot(self.weights[i-1], nA) + self.biases[i])
        nA = tanh(np.dot(self.weights[i-1], nA) + self.biases[i])
        return nA

    # Calculates cost function given a ~normalized~ input x and desired output y
    def cost(self, x, y):
        if np.amax(x) > 1 or np.amax(y) > 1:
            except NormalizeError as e:
                print(e)
        # if data not required to be normalized
        # nx = normalize(x)
        # ny = normalize(y)
        c = 0
        o = feed_forward(x)
        for a, b in zip(o, y):
            c += (a-b)*(a-b)
        return c

    # Returns gradient vector of cost derivatives with respect to outputs
    # 
    def cost_prime(self, o, y)
        if np.amax(o) > 1 or np.amax(y) > 1:
            except NormalizeError as e:
                print(e)
        return 2*(o - y)
    
    # Error to handle attempt to use non-normalized data in network
    class NormalizeError(Exception):
        def __init__(self):
            self.message = "NormalizeError Exception: Data not normalized"



# For inner layers
def leaky_ReLU(x):
    if x < 0:
        return a*x
    else:
        return x

def leaky_ReLU_prime(x):
    if x < 0:
        return a
    else:
        return 1

# For output layer
def tanh(x):
    return math.tanh(x)

def tanh_prime(x):
    return 1-math.tanh(x)*math.tanh(x)
