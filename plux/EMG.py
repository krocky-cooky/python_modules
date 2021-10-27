import os,sys
sys.path.append(os.path.dirname(__file__))

import numpy as np 
import matplotlib.pyplot as plt
import argparse
import json

from utils.parameter import HEADER_OFFSET,FOOTER_OFFSET,FILE_INFO_INDEX,TXTFILE_PATH,RANGE_MV
from general import PluxData


#def argparser():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--path',default = TXTFILE_PATH)
#    args = parser.parse_args()

#    return args 





def parse_EMG_output(file_path = TXTFILE_PATH):
    data = PluxData(file_path)
    ret = dict()
    for key,val in data.get_data().items():
        if val['sensor'] != 'EMG':
            raise Exception('there is a data other than EMG')
        ret[key] = val['data']

    return ret,data.data_info


def get_EMG_RMS(
    file_path = TXTFILE_PATH,
    range_ms = 100,
    smoothing_mode = "same"
):
    data = PluxData(file_path)
    ret = data.get_EMG_RMS(
        range_ms = range_ms,
        smoothing_mode = smoothing_mode
        )
    return ret 


