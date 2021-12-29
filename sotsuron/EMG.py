import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import pandas as pd 
import matplotlib.pyplot as plt 
from plux.EMG import get_EMG_RMS,get_EMG_raw

def plot_on_condition(
    csv_file,
    emg = ['外腹斜筋','内腹斜筋'],
    pillow = ['傾斜なし','傾斜30','傾斜45','傾斜60'],
    rollover = ['上肢先行','下肢先行','膝立','横向き上肢先行','横向き下肢先行'],
    channel = ['CH1','CH2','CH3','CH4'],
    is_raw = False,
    ymax = 0.15
):
    df = pd.read_csv(csv_file)
    df_extracted = df[
        (df['寝返り型'].isin(rollover)) &
        (df['枕形状'].isin(pillow)) &
        (df['筋電'].isin(emg)) & 
        (df['チャンネル'].isin(channel))
    ]
    ncol = df_extracted.shape[0]
    fig = plt.figure(figsize = (10,ncol*5))
    for i,(index,row) in enumerate(df_extracted.iterrows()):
        path = row['絶対パス']
        data = None 

        if is_raw:
            data = get_EMG_raw(path)
        else:
            data = get_EMG_RMS(path)

        data = data[row['チャンネル']]
        ax = plt.subplot2grid((ncol,1),(i,0))
        ax.plot(data)
        if is_raw:
            ax.set_ylim(-ymax,ymax)
        else:
            ax.set_ylim(0,ymax)
        title = '{}/{}/{}/{}/{}回目/最大値:{}'.format(
            row['筋電'],
            row['枕形状'],
            row['寝返り型'],
            row['チャンネル'],
            int(row['id']+1),
            row['最大値']
        )
        ax.set_title(title,fontname = 'MS Gothic')
    plt.plot()

        


