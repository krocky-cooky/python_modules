import numpy as np


class Normal:
    def __init__(self,weight,bias,learning_rate):
        self.weight = weight
        self.bias = bias
        self.learning_rate = learning_rate
        self.delta = None
        self.input = None


    def update(self):
        self.weight -= self.learning_rate/self.delta.shape[0]*np.dot(self.input.T,self.delta)
        self.bias -= self.learning_rate/self.delta.shape[0]*np.sum(self.delta,axis=0)
        return (self.weight,self.bias)

class Momentum:
    def __init__(self,weight,bias,learning_rate,mu):
        self.weight = weight
        self.bias = bias
        self.learning_rate = learning_rate
        self.delta = None
        self.input = None
        self.delta_weight = np.zeros_like(self.weight)
        self.delta_bias = np.zeros_like(self.bias)
        self.mu = mu


    def update(self):
        self.delta_weight = self.mu*self.delta_weight-self.learning_rate/self.delta.shape[0]*np.dot(self.input.T,self.delta)
        self.delta_bias = self.mu*self.delta_bias -self.learning_rate/self.delta.shape[0]*np.sum(self.delta,axis=0)
        self.weight += self.delta_weight
        self.bias += self.delta_bias
        return (self.weight,self.bias)