import cv2
import os
import pandas as pd
from get_landmarks import get_landmarks

wlasl = pd.read_csv('data/WSASL_100/WLASL_100.csv', index_col = 0)

path_in = 'data/WSASL_100/videos/'
path_out = 'data/WSASL_100/landmarks/'

try:
    os.mkdir(path_out)
except:
    pass

n_files = wlasl.shape[0]

n = 0
for index, row in wlasl.iterrows():
    n = n + 1
    try:
        os.mkdir(path_out + row['gloss'])
    except:
        pass
       
    file_in = path_in + row['file']
    file_out = path_out + row['gloss'] + "/" + row['file'].strip('.mp4') + '.npy'
    
    print(n/n_files)
    get_landmarks(file_in, file_out, display = False)
