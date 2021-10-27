import os,sys
sys.path.append(os.path.dirname(__file__))

import numpy as np 
import matplotlib.pyplot as plt 
import json 
import re 


from utils.parameter import HEADER_OFFSET,FOOTER_OFFSET,FILE_INFO_INDEX
from utils.funcs import convertEMG16bittoVAL


class PluxData(object):
    def __init__(self,file_path):
        self.file_path = file_path

        with open(file_path,'r') as f:
            txt = f.read()
            file_data = txt.split('\n')

        data_section = file_data[HEADER_OFFSET:-FOOTER_OFFSET]
        header_data = json.loads(file_data[FILE_INFO_INDEX][1:])
        self.device_name = list(header_data.keys())[0]
        self.data_info = header_data[self.device_name]
        self.sampling_rate = float(self.data_info['sampling rate'])
        self.labels = self.data_info['label']
        self.sensors = self.data_info['sensor']
        self.resolutions = self.data_info['resolution']
        num_col = len(self.labels)
        self.result_data = np.array('\t'.join(data_section).split(),dtype = np.float).reshape((-1,num_col+2)).T[2:]
        self.data = dict()
        
        for label,sensor,resolution,arr in zip(self.labels,self.sensors,self.resolutions,self.result_data):
            self.data[label] = dict()
            self.data[label]['sensor'] = sensor

            if sensor == 'EMG':
                self.data[label]['data'] = np.array(list(map(convertEMG16bittoVAL,arr,[resolution for i in range(arr.shape[0])])))
            else:
                raise Exception('invalid sensor type')

        
    def get_data(self):
        return self.data

    def get_EMG_RMS(
        self,
        range_ms = 100,
        smoothing_mode = 'same'
    ):
        self.EMG_RMS = dict()
        range_frame = int(range_ms*self.sampling_rate / 1000)
        conv_base = np.ones(range_frame) / range_frame

        for label,dic in self.data.items():
            if dic['sensor'] != 'EMG':
                continue 

            arr = dic['data']
            val_square = arr*arr 
            RMS_output = np.sqrt(np.convolve(
                val_square,
                conv_base,
                mode = smoothing_mode 
            ))
            self.EMG_RMS[label] = RMS_output 

        return self.EMG_RMS 
            
            



            


         
