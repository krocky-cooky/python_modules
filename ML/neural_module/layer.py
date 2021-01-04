import os,sys
sys.path.append(os.path.dirname(__file__))

import math
import numpy as np
import scipy as sp
from activations import *
from optimizer import Normal,Momentum



       
class Layer:
    def __init__(
        self,
        input_size,
        output_size,
        activation,
        learning_rate,
        optimize_initial_weight,
        optimizer,
        mu
    ):
        self.input = None
        self.output = None
        bias = np.zeros((1,output_size))
        weight = None
        if optimize_initial_weight:
            weight = np.random.randn(input_size,output_size)/math.sqrt(input_size)
        else:
            weight = 0.01*np.random.randn(input_size,output_size)

        if activation == 'identity':
            self.activation = Identity()
        elif activation == 'relu':
            self.activation = Relu()
        elif activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'softmax':
            self.activation = Softmax()
        else:
            raise Exception('活性化関数が正しく指定されていません。')

        if optimizer == 'normal':
            self.optimizer = Normal(
                weight = weight,
                bias = bias,
                learning_rate = learning_rate
            )
        elif optimizer == 'momentum':
            self.optimizer = Momentum(
                weight = weight,
                bias = bias,
                learning_rate = learning_rate,
                mu = mu
            )
        else :
            raise Exception('正しいoptimizerではありません')

        self.inner_layer = [self.optimizer,self.activation]
    
    def forward(self,input):
        pass

    def __call__(self,input):
        return self.forward(input)
    

class hiddenAndOutputLayer(Layer):
    def forward(self,input):
        self.input = input
        output = input
        for layer in self.inner_layer:
            output = layer(output)
        self.output = output
        return output

    def backward(self,delta):
        for layer in reversed(self.inner_layer):
            delta = layer.backward(delta)
        return delta

    

class inputLayer(Layer):
    def __init__(self):
        self.input = None
        self.inner_layer = None

    def forward(self,input):
        self.input = input
        return input

    def backward(self,delta):
        return delta

