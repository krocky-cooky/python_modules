import numpy as np

class Loss:
    def __init__(self):
        pass
    def forward(self,y,t):
        pass

    def __call__(self,y,t):
        return self.forward(y,t)

class SumSquare(Loss):
    def forward(self,y,t):
        loss = np.sum((y-t)**2)/2/y.shape[0]
        return loss

    def backward(self,y,t):
        return y-t

class CrossEntropy(Loss):
    def forward(self,y,t):
        d = 1e-7
        loss = np.sum(t*np.log(y+d))
        return loss

    def backward(self,y,t):
        ret = y-t
        ret = ret/y
        ret = ret/(1-y)
        return ret