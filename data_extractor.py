import numpy as np 
import pandas as pd
import os



images = []
mask = []
for i in range(1,11):
    data_dir=f'data/Class{i}/Train'
    data_dir_lab=f'/dataClass{i}/Train/Label'
    
    for i in os.listdir(data_dir_lab):
        if i.endswith(".PNG"):
            mask.append(data_dir_lab+'/'+i)
            temp=i.replace("_label",'')
            images.append(data_dir+'/'+temp)


for i in range(1,11):
    data_dir=f'data/Class{i}/Test'
    data_dir_lab=f'data/Class{i}/Test/Label'
    
    for i in os.listdir(data_dir_lab):
        if i.endswith(".PNG"):
            mask.append(data_dir_lab+'/'+i)
            temp=i.replace("_label",'')
            images.append(data_dir+'/'+temp)

data=pd.DataFrame({'image': images, 'mask':mask}, columns=['image', 'mask'])

data.to_csv('data/paths',index=False)
