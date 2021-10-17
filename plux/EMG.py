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



def parse_EMG_output(file_path = TXTFILE_PATH):
    #args = argparser()
    
    with open(file_path,'r') as f:
        txt = f.read()
        file_data = txt.split('\n')


    emg_data = file_data[HEADER_OFFSET:-FOOTER_OFFSET]
    header_data = json.loads(file_data[FILE_INFO_INDEX][1:])
    #print(header_data)
    device_name = list(header_data.keys())[0]
    data_info = header_data[device_name]
    
    sampling_rate = float(data_info['sampling rate'])
    column = data_info['column']
    resolution = data_info['resolution']

    output_data = dict()

    
    for i,c in enumerate(column):
        if i < 2:
            continue
        data_array = list()
        for d in emg_data:
            splited = d.split('\t') 
            emg_converted = converteMG16bittoVAL(
                emg = float(splited[i]),
                resolution = resolution[i-2],
            )
            data_array.append(emg_converted)
        
        output_data[c] = np.array(data_array)


    return output_data,data_info


def get_EMG_RMS(
    file_path = TXTFILE_PATH,
    range_ms = 100
):
    output_data,data_info = parse_EMG_output(file_path)
    keys = output_data.keys()
    RMS_data = dict()

    sampling_rate = data_info['sampling rate']
    range_frame = int(range_ms*sampling_rate / 1000)
    conv_base = np.ones(range_frame) / range_frame

    for key in keys:
        arr = output_data[key]
        val_square = arr*arr
        RMS_output = np.sqrt(np.convolve(
            val_square,
            conv_base,
            mode = "same"
            ))
        RMS_data[key] = RMS_output

    return RMS_data
    