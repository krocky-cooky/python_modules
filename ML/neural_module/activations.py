import numpy as np
import math


class ActivationLayer:
    def __init__(self):
        self.v = None
        self.y = None

    def forward(self,input):
        pass

    def backward(self,delta):
        pass
    
    def __call__(self,x):
        return self.forward(x)

class Sigmoid(ActivationLayer):
    def forward(self,input):
        self.y = 1/(1+np.exp(-x))
        return self.y
    
    def backward(self,delta):
        ret = self.y*(1-self.y)*delta
        return ret

class Relu(ActivationLayer):
    def forward(self,input):
        self.v = input
        self.y = np.maximum(0,input)
        return self.y

    def backward(self,delta):
        mask = (self.v > 0)
        res = np.zeros_like(self.v)
        #print(mask)
        res[mask] = 1
        return res*delta

class Identity(ActivationLayer):
    def forward(self,input):
        self.v = input
        self.y = input
        return self.y
    
    def backward(self,delta):
        return delta

class Softmax(ActivationLayer):
    def forward(self,input):
        self.v = input
        c = np.max(a,axis=1,keepdims=True)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a,axis = 1,keepdims=True)
        self.y = exp_a/sum_exp_a
        return self.y

    def backward(self,delta):
        dx = self.y*delta
        dsum = np.sum(self.y*delta,axis=1,keepdims=True)
        dx -= self.y*dsum
        return dx

