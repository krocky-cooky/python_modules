import os,sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
from sklearn import metrics

def r2_score(y,t):
    y_tmp = y.flatten()
    t_tmp = t.flatten()
    return metrics.r2_score(y_tmp,t_tmp)

def rmse(y,t):
    y_tmp = y.flatten()
    t_tmp = t.flatten()
    return np.sqrt(metrics.mean_squared_error(y_tmp, y_tmp))

def mae(y,t):
    y_tmp = y.flatten()
    t_tmp = t.flatten()
    return metrics.mean_absolute_error(y_obs, y_pred)


