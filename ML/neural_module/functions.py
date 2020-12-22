import numpy as np
import math

def softmax(a,div):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

def sigmoid(x,div):
    if div:
        tmp = sigmoid(x,False)
        return tmp*(1-tmp)
    else:
        return 1/(1+np.exp(-x))


def Relu(x,div):
    if div:
        mask = (x > 0)
        res = np.zeros_like(x)
        #print(mask)
        res[mask] = 1
        return res
    else:
        return np.maximum(0,x)

def identity(x,div = False):
    if div:
        return 1
    else :
        return x


def euler_loss(y,t,div = False):
    if div:
        return y-t
    else:
        loss = np.sum((y-t)**2)/2
        
        return loss

def cross_entropy_loss(y,t,div = False):
    if div:
        ret = y-t
        ret = ret/y
        ret = ret/(1-y)
        return ret
    else:
        delta = 1e-7
        loss = -np.sum(t*np.log(y + delta))
        return loss

