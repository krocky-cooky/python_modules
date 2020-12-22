import os,sys
sys.path.append(os.path.dirname(__file__))

import math
import numpy as np
import scipy as sp
from functions import softmax,Relu,sigmoid,identity
from optimizer import Normal,Momentum

class hiddenLayer:
    def __init__(
        self,
        input_size,
        output_size,
        activation = 'Relu',
        learning_rate=0.001,
        optimize_initial_weight = True,
        optimizer = 'normal',
        mu = 0.5,
    ):
        self.bias = np.zeros((1,output_size))
        self.weight = None
        if optimize_initial_weight:
            self.weight = np.random.randn(input_size,output_size)/math.sqrt(input_size)
        else:
            self.weight = 0.01*np.random.randn(input_size,output_size)
        self.which_activation = activation
        self.learning_rate = learning_rate
        self.v = None
        self.y = None
        self.delta = None
        self.input = None
        self.optimizer_name = optimizer
        if optimizer == 'normal':
            self.optimizer = Normal(
                weight = self.weight,
                bias = self.bias,
                learning_rate = learning_rate
            )
        elif optimizer == 'momentum':
            self.optimizer = Momentum(
                weight = self.weight,
                bias = self.bias,
                learning_rate = learning_rate,
                mu = mu
            )
        else :
            raise Exception('正しいoptimizerではありません')


    def process(self,input):
        self.optimizer.input = input
        self.v = np.dot(input,self.weight) + self.bias
        self.y = self.activation(self.v)
        return self.y

    def activation(self,input,div=False):
        name = self.which_activation
        if name == 'softmax':
            return softmax(input,div)
        elif name == 'Relu':
            return Relu(input,div)
        elif name == 'sigmoid':
            return sigmoid(input,div)
        elif name == 'identity':
            return identity(input,div)
        else:
            raise Exception('活性化関数が正しく指定されていません。')

    def update_delta(self,dif):
        self.optimizer.delta = self.activation(self.v,div=True)*dif
        


    def update_weight(self):
        self.weight,self.bias = self.optimizer.update()
        #x = self.learning_rate/self.delta.shape[0]*np.dot(self.input.T,self.delta)
        #self.weight -= x
        #self.bias -= self.learning_rate/self.delta.shape[0]*np.sum(self.delta,axis=0)




    def send_backword(self):
        dif = np.dot(self.optimizer.delta,self.weight.T)
        return dif




class inputLayer:
    def __init__(self,size):
        self.size = size

    def process(self,input):
        return input



class outputLayer:
    def __init__(
        self,
        input_size,
        output_size,
        activation = 'identity',
        learning_rate=0.001,
        optimize_initial_weight = True,
        optimizer = 'normal',
        mu = 0.5
    ):
        self.bias = np.zeros((1,output_size))
        self.weight = None
        if optimize_initial_weight:
            self.weight = np.random.randn(input_size,output_size)/math.sqrt(input_size)
        else:
            self.weight = 0.01*np.random.randn(input_size,output_size)

        self.which_activation = activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        if optimizer == 'normal':
            self.optimizer = Normal(
                weight = self.weight,
                bias = self.bias,
                learning_rate = learning_rate
            )
        elif optimizer == 'momentum':
            self.optimizer = Momentum(
                weight = self.weight,
                bias = self.bias,
                learning_rate = learning_rate,
                mu = mu
            )
        else :
            raise Exception('正しいoptimizerではありません')

    def process(self,input):
        self.optimizer.input = input
        self.v = np.dot(input,self.weight) + self.bias
        self.y = self.activation(self.v)
        return self.y

    def activation(self,input,div = False):
        name = self.which_activation
        if name == 'softmax':
            return softmax(input,div)
        elif name == 'Relu':
            return Relu(input,div)
        elif name == 'sigmoid':
            return sigmoid(input,div)
        elif name == 'identity':
            return identity(input,div)
        else:
            raise Exception('活性化関数が正しく指定されていません。')

    def update_delta(self,dif):
        self.optimizer.delta = self.activation(self.v,div=True)*dif

    def update_weight(self):
        self.weight,self.bias = self.optimizer.update()
        #self.weight -= self.learning_rate/self.delta.shape[0]*np.dot(self.input.T,self.delta)
        #self.bias -= self.learning_rate/self.delta.shape[0]*np.sum(self.delta,axis=0)


    def send_backword(self):
        dif = np.dot(self.optimizer.delta,self.weight.T)
        return dif