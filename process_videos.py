import cv2
import os
import pandas as pd
from get_landmarks import get_landmarks
from video_augmentation import *

import gc

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
    # extracting landmarks of original video       
    file_in = path_in + row['file']
    file_out = path_out + row['gloss'] + "/" + row['file'].strip('.mp4') + '.npy'
    
    print(file_out)
    
    get_landmarks(file_in, file_out, display = False)
    
    
    # adding 10 random transformation of the original video for data augmentation
    for k in range(10):
        # randomly transform video and saving in a temp file
        va = video_augmentation(file_in, 'temp.mp4')
        # updating file out path
        file_out = path_out + row['gloss'] + "/" + row['file'].strip('.mp4') + '_' + str(k) + '.npy'
     
        print(file_out)
        
        # getting landmarks from the temp video file
        get_landmarks('temp.mp4', file_out, display = False)
        
        
    print(n/n_files)
    
    gc.collect()
    
os.remove('temp.mp4')
