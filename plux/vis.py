import os,sys 
import matplotlib.pyplot as plt 
import numpy as np 


def visualize_pulse(
    data,
    len_col = 1000
):
    start = 0
    end = 1000
    d = data
    ncol = int(len(d)/len_col)
    if ncol%len_col !=  0:
        ncol += 1
    fig = plt.figure(figsize=(10,5*ncol))
    index = 0
    ymin = np.min(d)
    ymax = np.max(d)
    range_data = ymax-ymin

    while True:
        ax = plt.subplot2grid((ncol,1),(index,0))
        #print('{} to {}'.format(start,min(end,d.shape[0])))
        ax.plot(d[start:min(end,len(d))])
        ax.set_title('{} to {}'.format(start,min(end,len(d))))
        ax.set_ylim(ymin-range_data/10,ymax+range_data/10)
        if end > len(d):
            break
        start += 1000
        end += 1000
        index += 1
    
    plt.show()