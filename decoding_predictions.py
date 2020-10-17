import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage
from skimage.io import imread,imshow
import glob
import pandas as pd
import pickle
import time
from sklearn.preprocessing import MinMaxScaler


df=pd.read_csv(r'Results/E3D3/1.csv')

with open('Data/imputed_valid_data.pkl', 'rb') as picklefile:
    valid_data=pickle.load(picklefile)

with open('Models/DR/trainedPCA.pkl', 'rb') as picklefile:
    pca=pickle.load(picklefile)


daf=df.drop(['date'],axis='columns')

fpred=pca.inverse_transform(daf.values)

ad=np.reshape(fpred,(len(fpred),256,272,1))

for i,date in enumerate(df['date'].values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    img = scaler.fit_transform(abs(fpred[i]).reshape(-1,1))
    img=np.reshape(img,(256,272))
    ad[i]=img.reshape((256,272,1))
    matplotlib.image.imsave('Final Pred/E3D3/'+date+'.jpg', img,cmap='gray')
    time.sleep(0.1)
    # break

valid_data = valid_data[45:-6]


## Calculating Scores

from skimage.measure import compare_ssim as ssim

def SSIM(y_true, y_pred):
    return tf.reduce_mean(ssim(y_true, y_pred,multichannel=True))

def PSNR(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))        

from sklearn.metrics import mean_absolute_error,r2_score
print(mean_absolute_error(valid_data.reshape((len(valid_data),256*272)),ad.reshape((len(valid_data),256*272))))

ps = tf.image.psnr(valid_data,ad, 1)

print('PSNR score : ',ps.numpy().mean())

ss=[]
for i in range(0,len(valid_data)):
	ss.append(ssim(valid_data[i],ad[i],multichannel=True))    

print('SSIM score :',sum(ss)/len(ss)