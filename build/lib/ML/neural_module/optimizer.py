import numpy as np


class Optimizer:
    def __init__(self,weight,bias,learning_rate,mu = None):
        self.weight = weight
        self.bias = bias
        self.learning_rate = learning_rate
        self.input = None
        self.output = None
        self.mu = mu

    def forward(self,input):
        self.input = input
        self.output = np.dot(input,self.weight) + self.bias
        return self.output

    def __call__(self,input):
        return self.forward(input)


class Normal(Optimizer):
    def backward(self,delta):
        self.weight -= self.learning_rate/delta.shape[0]*np.dot(self.input.T,delta)
        self.bias -= self.learning_rate/delta.shape[0]*np.sum(delta,axis=0)
        n_delta = np.dot(delta,self.weight.T)
        return n_delta 

class Momentum(Optimizer):
    def __init__(self,weight,bias,learning_rate,mu = None):
        super().__init__(weight,bias,learning_rate,mu)
        self.delta_weight = np.zeros_like(self.weight)
        self.delta_bias = np.zeros_like(self.bias)


    def backward(self,delta):
        self.delta_weight = self.mu*self.delta_weight-self.learning_rate/delta.shape[0]*np.dot(self.input.T,delta)
        self.delta_bias = self.mu*self.delta_bias -self.learning_rate/delta.shape[0]*np.sum(delta,axis=0)
        self.weight += self.delta_weight
        self.bias += self.delta_bias
        n_delta = np.dot(delta,self.weight.T)
        return n_delta