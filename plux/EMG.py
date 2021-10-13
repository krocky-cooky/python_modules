import os,sys
import numpy as np 
import matplotlib.pyplot as plt
import argparse
import json

TXTFILE_PATH = ''
HEADER_OFFSET = 3
FOOTER_OFFSET = 1
FILE_INFO_INDEX = 1
RANGE_MV = 1.5

#def argparser():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--path',default = TXTFILE_PATH)
#    args = parser.parse_args()

#    return args 

def converteMG16bittoVAL(emg,resolution):
    value = ((emg - 2**(resolution-1)) / 2**(resolution-1) * RANGE_MV)
    return value



def parse_EMG_output(file_path = TXTFILE_PATH)
    #args = argparser()
    
    with open(file_path,'r') as f:
        txt = f.read()
        file_data = txt.split('\n')


    emg_data = file_data[HEADER_OFFSET:-FOOTER_OFFSET]
    data_info = json.loads(file_data[FILE_INFO_INDEX][1:])
    
    sampling_rate = data_info['sampling rate']
    column = data_info['column']
    resolution = data_info['resolution']

    output_data = dict()
    for key in column:
        output_data[key] = list()

    for d in emg_data:
        splited = d.split('\t')
        for i,c in enumerate(column):
            if i < 2:
                continue
            emg_converted = converteMG16bittoVAL(
                emg = splited[i],
                resolution = resolution[i-2],
            )
            output_data[c].append(emg_converted)

    return output_data