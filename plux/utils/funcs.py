import os,sys 
sys.path.append(os.path.dirname(__file__))

from parameter import RANGE_MV

def convertEMG16bittoVAL(emg,resolution):
    value = ((emg - 2**(resolution-1)) / 2**(resolution-1) * RANGE_MV)
    return value