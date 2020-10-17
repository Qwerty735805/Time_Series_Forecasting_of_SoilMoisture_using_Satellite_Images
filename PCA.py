import pandas as pd
import numpy as np
import skimage
from skimage.io import imread
from tqdm import tqdm
import os
import pickle

with open('Data/imputed_valid_data.pkl', 'rb') as picklefile:
    valid_data=pickle.load(picklefile)

with open('Data/imputed_train_data.pkl', 'rb') as picklefile:
    train_data=pickle.load(picklefile)

from sklearn.decomposition import PCA

pca = PCA(0.995)
pca.fit(train_data.reshape(len(train_data),256*272))


with open('Models/trainedPCA.pkl', 'wb') as picklefile:
    pickle.dump(pca, picklefile)

        
