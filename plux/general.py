import os,sys
sys.path.append(os.path.dirname(__file__))

import numpy as np 
import matplotlib.pyplot as plt 
import json 
import re 

from utils.parameter import HEADER_OFFSET,FOOTER_OFFSET,FILE_INFO_INDEX
from utils.funcs import convertEMG16bittoVAL



class PluxData(object):
    """
    Attributes
    -----
    device_name: str
        the name of bioSignalPlux device

    data_info: dict
        the row infomation dictionary of the data

    sampling_rate: int
        sampling rate 

    labels: list
        list of the name of the channel labels
        eg) ['CH1' , 'CH2'] 

    sensors: list 
        list of the each sensor name

    resolutons: list
        list of each sensors' resolution

    result_data: ndarray
        ndarray of the row data got from the text data
    
    data: dict 
        dictionary of the processed data
    """

    def __init__(self,file_path):
        self.file_path = file_path

        with open(file_path,'r') as f:
            txt = f.read()
            file_data = txt.split('\n')
        
        if not 'OpenSignals' in file_data[0]:
            raise Exception('OpenSignalsにより生成されたデータではありません')
            
        data_section = file_data[HEADER_OFFSET:-FOOTER_OFFSET]
        header_data = json.loads(file_data[FILE_INFO_INDEX][1:])
        self.device_name = list(header_data.keys())[0]
        self.data_info = header_data[self.device_name]
        self.sampling_rate = float(self.data_info['sampling rate'])
        self.labels = self.data_info['label']
        self.sensors = self.data_info['sensor']
        self.resolutions = self.data_info['resolution']
        num_col = len(self.labels)
        self.result_data = np.array('\t'.join(data_section).split(),dtype = np.float32).reshape((-1,num_col+2)).T[2:]
        self.data = dict()
        
        for label,sensor,resolution,arr in zip(self.labels,self.sensors,self.resolutions,self.result_data):
            self.data[label] = dict()
            self.data[label]['sensor'] = sensor

            if sensor == 'EMG':
                self.data[label]['data'] = np.array(list(map(convertEMG16bittoVAL,arr,[resolution for i in range(arr.shape[0])])))
            else:
                raise Exception('invalid sensor type')

        
    def get_data(self):
        """
        params
        -----
        None

        Returns 
        -----
        data: dict
            processed data
        """
        return self.data

    def get_EMG_RMS(
        self,
        range_ms = 100,
        smoothing_mode = 'same',
        filt = None
    ):
        """
        params
        -----
        range_ms: int
            smoothing range
            default: 100

        smoothing_mode: str 
            smoothing mode of np.convolve
            default:  'same'

        Returns
        -----
        EMG_RMS: dict
            the data processed to RMS
        """
        self.EMG_RMS = dict()
        range_frame = int(range_ms*self.sampling_rate / 1000)
        conv_base = np.ones(range_frame) / range_frame

        for label,dic in self.data.items():
            if dic['sensor'] != 'EMG':
                continue 

            arr = dic['data']
            if filt is not None:
                arr = filt(arr)
            val_square = arr*arr 
            RMS_output = np.sqrt(np.convolve(
                val_square,
                conv_base,
                mode = smoothing_mode 
            ))
            self.EMG_RMS[label] = RMS_output 

        return self.EMG_RMS 

    def get_EMG_raw(self):
        """
        params
        -----

        Returns
        -----
        EMG_raw: dict
            the raw data of EMG
        """
        self.EMG_raw = dict()
        
        for label,dic in self.data.items():
            if dic['sensor'] != 'EMG':
                continue 

            arr = dic['data']
            self.EMG_raw[label] = arr 

        return self.EMG_raw 
            



            


         
