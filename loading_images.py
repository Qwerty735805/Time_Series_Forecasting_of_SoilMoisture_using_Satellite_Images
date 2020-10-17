import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import skimage
from skimage.io import imread
from tqdm import tqdm
from datetime import date, timedelta

train_data,valid_data = [],[]
flag=0
missing = np.full([256,272,1], np.nan)

train_start = date(2015,4,4)
train_end   = date(2020,3,29)
valid_start = date(2020,5,24)
valid_end   = date(2020,10,9)
one_day = timedelta(days=1)

for filename in tqdm(os.listdir('Data/tif files')):
    # print(filename)
    name= str(filename[-12:-8])+'-'+str(filename[-8:-6])+'-'+str(filename[-6:-4])
    current = date(int(name[0:4]),int(name[5:7]),int(name[8:]))
    if flag==0:
        if current != train_start:
            train_data.extend([missing]*int((current - train_start).days))
            train_start = current
    else:
        if current != valid_start:
            valid_data.extend([missing]*int((current - valid_start).days))
            valid_start = current
    img = imread('Data/tif files/'+filename)
    img = np.expand_dims(img,axis=-1)
    img = np.asarray(img)
    if flag==0:
        train_data.append(img)
        train_start += one_day    
    else :
        valid_data.append(img)
        valid_start += one_day
    if filename == 'SWI_SMAP_I_20200327_20200329.tif':
        flag = 1
train_data=np.asarray(train_data)
valid_data=np.asarray(valid_data)
data = np.concatenate((train_data,valid_data),axis=0)
print(train_data.shape)
print(valid_data.shape)
print(data.shape)

with open('Data/data.pkl', 'wb') as picklefile:
    pickle.dump(data, picklefile)
with open('Data/train_data.pkl', 'wb') as picklefile:
    pickle.dump(train_data, picklefile)
with open('Data/valid_data.pkl', 'wb') as picklefile:
    pickle.dump(valid_data, picklefile)