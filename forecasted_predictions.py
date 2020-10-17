import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tqdm import tqdm


model_e1d1.load_weights('Models/TSF/E1D1.h5')
model_e2d2.load_weights('Models/TSF/E2D2.h5')
model_e2d1.load_weights('Models/TSF/E2D1.h5')
model_e3d2.load_weights('Models/TSF/E3D2.h5')
model_e3d3.load_weights('Models/TSF/E3D3.h5')

X_train, y_train = split_sequences(train_data.values,n_steps_in, n_steps_out)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))

X_valid, y_valid = split_sequences(valid_data.values,n_steps_in, n_steps_out)
X_valid = X_valid.reshape((X_valid.shape[0], X_train.shape[1],n_features))
y_valid = y_valid.reshape((y_valid.shape[0], y_train.shape[1], n_features))
 

pred=model_e3d3.predict(X_valid)

for i in range(1,588):
    scaler = scalers['scaler'+str(i)]
    pred[:,:,i-1]=scaler.inverse_transform(pred[:,:,i-1])
#     y_train[:,:,i-1]=scaler.inverse_transform(y_train[:,:,i-1])
    y_valid[:,:,i-1]=scaler.inverse_transform(y_valid[:,:,i-1])


daf1=valid_df.iloc[45:-6].reset_index(drop=True)
daf2=valid_df.iloc[46:-5].reset_index(drop=True)
daf3=valid_df.iloc[47:-4].reset_index(drop=True)
daf4=valid_df.iloc[48:-3].reset_index(drop=True)
daf5=valid_df.iloc[49:-2].reset_index(drop=True)
daf6=valid_df.iloc[50:-1].reset_index(drop=True)
daf7=valid_df.iloc[51:].reset_index(drop=True)
daf = [daf1,daf2,daf3,daf4,daf5,daf6,daf7]


for j in range(1,588):
    # print(j)
    for i in range(0,7):
        # print('\t',i)
        daf[i]['unit'+str(j)]=pred[:,i,j-1]    

for i,df in enumerate(daf):
    df.to_csv(r'Results/E3D3/'+str(i+1)+'.csv', index = False)            