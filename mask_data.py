import pandas as pd
import numpy as np
import skimage
from skimage.io import imread
from tqdm import tqdm
import os
import pickle

with open('Data/train_data.pkl', 'rb') as picklefile:
    train_data=pickle.load(picklefile)
with open('Data/valid_data.pkl', 'rb') as picklefile:
    valid_data=pickle.load(picklefile)

train_data_nan = np.isnan(train_data)

mask = train_data_nan[0]

for i in train_data_nan:
	mask = mask & i

mask_train_data,mask_valid_data = train_data,valid_data
for i,a in enumerate(train_data):
    a[mask==1]=0 
    mask_train_data[i]=a

mask_valid_data = mask_valid_data
for i,a in enumerate(valid_data):
    a[mask==1]=0 
    mask_valid_data[i]=a

mask_data=np.concatenate((mask_train_data,mask_valid_data),axis=0)
with open('Data/masked_data.pkl', 'wb') as picklefile:
    pickle.dump(mask_data, picklefile)
with open('Data/masked_train_data.pkl', 'wb') as picklefile:
    pickle.dump(mask_train_data, picklefile)
with open('Data/masked_valid_data.pkl', 'wb') as picklefile:
    pickle.dump(mask_valid_data, picklefile)
