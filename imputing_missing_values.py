import pandas as pd
import numpy as np
import skimage
from skimage.io import imread
from tqdm import tqdm
import os
import pickle


with open('Data/masked_valid_data.pkl', 'rb') as picklefile:
    mask_valid_data=pickle.load(picklefile)

with open('Data/masked_train_data.pkl', 'rb') as picklefile:
    mask_train_data=pickle.load(picklefile)

mask_train_data = np.reshape(mask_train_data,(len(mask_train_data),256*272))
mask_valid_data = np.reshape(mask_valid_data,(len(mask_valid_data),256*272))

imputed_train_data,imputed_valid_data = mask_train_data,mask_valid_data

for i in tqdm(range(256*272)):
    series= pd.Series(imputed_train_data[:,i])
    series = series.interpolate(method='linear')
    series = series.fillna(method='ffill')
    series = series.fillna(method='bfill')
    series = series.fillna(0)
    imputed_train_data[:,i]=series.values

for i in tqdm(range(256*272)):
    series= pd.Series(imputed_valid_data[:,i])
    series = series.interpolate(method='linear')
    series = series.fillna(method='ffill')
    series = series.fillna(method='bfill')
    series = series.fillna(0)
    imputed_valid_data[:,i]=series.values

imputed_train_data = imputed_train_data.reshape(len(imputed_train_data),256,272,1)
imputed_valid_data = imputed_valid_data.reshape(len(imputed_valid_data),256,272,1)
imputed_data = np.concatenate((imputed_train_data,imputed_valid_data),axis=0)

with open('Data/imputed_data.pkl', 'wb') as picklefile:
    pickle.dump(imputed_data, picklefile)

with open('Data/imputed_train_data.pkl', 'wb') as picklefile:
    pickle.dump(imputed_train_data, picklefile)

with open('Data/imputed_valid_data.pkl', 'wb') as picklefile:
    pickle.dump(imputed_valid_data, picklefile)