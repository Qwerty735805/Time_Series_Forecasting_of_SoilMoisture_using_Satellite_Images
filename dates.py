import os
from tqdm import tqdm
from datetime import date, timedelta
import pickle

train_dates=[]
valid_dates=[]
flag=0

for filename in tqdm(os.listdir('Data/tif files')):
    temp_date=str(filename[-12:-8])+'-'+str(filename[-8:-6])+'-'+str(filename[-6:-4])
    if flag==0:
    	train_dates.append(temp_date)
    else :
    	valid_dates.append(temp_date)
    if filename == 'SWI_SMAP_I_20200327_20200329.tif':
    	flag = 1
        
train_dates_list = [date(int(i[0:4]),int(i[5:7]),int(i[8:])) for i in train_dates]
valid_dates_list = [date(int(i[0:4]),int(i[5:7]),int(i[8:])) for i in valid_dates]


start_date = train_dates_list[0]
missing_train_dates=[]
one_day = timedelta(days=1)
while start_date < train_dates_list[-1]:
    if start_date not in train_dates_list:
        missing_train_dates.append(start_date)
    start_date += one_day
    
train_dates_list = [str(i) for i in train_dates_list]
missing_train_dates = [str(i) for i in missing_train_dates]
train_dates_list = train_dates_list + missing_train_dates

train_dates_list.sort()


start_date = valid_dates_list[0]
missing_valid_dates=[]
one_day = timedelta(days=1)
while start_date < valid_dates_list[-1]:
    if start_date not in valid_dates_list:
        missing_valid_dates.append(start_date)
    start_date += one_day
    
valid_dates_list = [str(i) for i in valid_dates_list]
missing_valid_dates = [str(i) for i in missing_valid_dates]
valid_dates_list = valid_dates_list + missing_valid_dates

valid_dates_list.sort()


with open('train_dates.pkl', 'wb') as picklefile:
    pickle.dump(train_dates_list, picklefile)
with open('valid_dates.pkl', 'wb') as picklefile:
    pickle.dump(valid_dates_list, picklefile)