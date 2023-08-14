# functions for video augmentation

import cv2
import numpy as np
from scipy import ndimage
import skvideo.io


def frame_augmentation(frame, mirror = False, angle = 0, shift = [0, 0, 0]):
    out_frame = ndimage.rotate(frame, angle, reshape = False, order = 0)
    out_frame = ndimage.shift(out_frame, shift, order = 0)
    if mirror:
        out_frame = cv2.flip(out_frame, 1)
    return(out_frame)
    
    
def video_augmentation(file_in, file_out = None, max_angle = 35, max_shift = 0.2, max_time_shift = 0.3):
    # applies random transformations to a video for data augmentation
    # file_in: input file
    # file_out: optional file name for writing transformed video
    # max_anlge: maximal rotation in degrees
    # max shift: maximal shift (in proportion of the image size)
    # time_shift: by how much time do we shift the video (as a proportion of the number of frames)
    
    # reading file
    cap = cv2.VideoCapture(file_in)
    # extracting resolution
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # drawing random transformations
    
    # mirroring T/F
    mirror = np.random.randint(0, 2)
    # rotation up to 35 degrees
    angle = np.random.uniform(-max_angle, max_angle)
    # shift
    shift_x = int(np.random.uniform(-max_shift, max_shift)*width)
    shift_y = int(np.random.uniform(-max_shift, max_shift)*height)
    
    # time shift:
    time_shift = int(np.random.uniform(-max_time_shift, max_time_shift) * n_frames)
    
    # creating a list to extract all frames
    frame_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret == True:
            frame_list.append(frame)
        else:
            break
    cap.release()
    
    # shifting the frame list in time
    if time_shift > 0:
        frame_list = frame_list[time_shift:]
    if time_shift < 0:
        frame_list = frame_list[:time_shift]
    # augmenting the video
    augmented_frame_list = [frame_augmentation(frame, mirror, angle, [shift_x, shift_y, 0]) for frame in frame_list]
    
    # write out
    if file_out:
        skvideo.io.vwrite(file_out, augmented_frame_list)
    
    return(frame_list)
    
    
