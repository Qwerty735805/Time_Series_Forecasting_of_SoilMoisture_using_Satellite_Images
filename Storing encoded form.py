import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread
from skimage.transform import resize
import glob
import pandas as pd
from tqdm import tqdm
import pickle


with open('Models/DR/trainedPCA.pkl', 'rb') as picklefile:
    pca=pickle.load(picklefile)
    
with open('Data/imputed_valid_data.pkl', 'rb') as picklefile:
    valid_data=pickle.load(picklefile)

with open('Data/imputed_train_data.pkl', 'rb') as picklefile:
    train_data=pickle.load(picklefile)    



columns=[]
for i in range(1,1 + pca.n_components_):
    columns.append('unit'+str(i))


train_df=pd.DataFrame(columns=columns)
valid_df=pd.DataFrame(columns=columns)


train_df.insert(0, "date",[])
valid_df.insert(0, "date",[])


flag = 0

for i,filename in tqdm(enumerate(os.listdir('Data/tif files'))):
    
#     print(i," ",filename)
    if flag==0:
        img = train_data[i]
    else :                           
        img = valid_data[i-len(train_data)]
    encoded_img = pca.transform(img.reshape(1,256*272))
    encoded_img=np.reshape(encoded_img,(604))
    temp_dict={}
    temp_dict['date']=str(filename[-12:-8])+'-'+str(filename[-8:-6])+'-'+str(filename[-6:-4])
    count=0
    for i in encoded_img:
        temp_dict['unit'+str(count+1)]=i
        count=count+1    
    if flag==0:
        train_df = train_df.append(temp_dict,ignore_index=True)
    else :                           
        valid_df = valid_df.append(temp_dict,ignore_index=True)
    if filename == 'SWI_SMAP_I_20200327_20200329.tif':
    	flag = 1

train_df.to_csv(r'imputed_train_encoded_data.csv', index = False)
valid_df.to_csv(r'imputed_valid_encoded_data.csv', index = False)




