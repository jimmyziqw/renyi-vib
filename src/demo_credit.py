import pandas as pd
import numpy as np
raw_df = pd.read_csv("D:\AFUNFOLDER\Renyi\input\data\loan\\application_data.csv")
raw_df.head()
from sklearn.preprocessing import LabelEncoder
raw_df["CODE_GENDER"] = LabelEncoder().fit_transform(raw_df["CODE_GENDER"])
df = raw_df.dropna(how="any")


target_cols = ["TARGET","CODE_GENDER"]
y = df[target_cols].astype("int64").to_numpy()
df= df.drop(["SK_ID_CURR"]+target_cols, axis=1)

cat_cols= df.select_dtypes(include=['bool','object']).columns
float64_cols = df.select_dtypes(include=['int64','float64']).columns
int64_cols = df.select_dtypes(include=['uint8']).columns

#https://www.kaggle.com/datasets/gauravduttakiit/loan-defaulter
df[float64_cols] = df[float64_cols].apply(lambda x: np.tanh(x+1)).astype("float32")
df[int64_cols] = df[int64_cols].astype('int32')
data = pd.get_dummies(df,columns=cat_cols,drop_first=True)
#print(data.dtypes)
#print(data.columns)

#print(type(y))
x = data.dropna().to_numpy()

#data = (x,y)
#print(data)
print(x.shape)
from renyi_vib import *
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y,
     test_size=0.2, random_state=44)
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    
])
#print(x_train.shape,y_train.shape)
train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
#print(f"train set size {len(train)}, validation set size {len(test)}")

train_dataloader = DataLoader(train, batch_size=batch_size,shuffle=True)
val_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)
idx,(x,y)= next(enumerate(train_dataloader))
#print('data_shape',x)
vib = FairVIB(216, 2, 128).to(device)
print("Device: ", device)
# Training
import time 
#from collections import defaultdict
start_time = time.time()
vib.fit(train_dataloader,val_dataloader)
# put Deep VIB into train mode 