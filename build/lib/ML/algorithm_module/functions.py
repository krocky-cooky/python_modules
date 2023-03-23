import numpy as np
import sys,os

def dist(a,b):
    ret = np.sum((a-b)**2,axis = 1)

    return ret

def r2_score(y,y_pred):
    up = np.sum((y-y_pred)**2)
    ave = np.sum(y/float(y.shape[0]))
    bottom = np.sum((y-ave)**2)

    return 1-up/bottom

