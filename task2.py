

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader,random_split,SubsetRandomSampler
from sklearn import preprocessing
from sklearn.model_selection import KFold
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(33 , 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 13)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
model=Net()





rop=['CUBE_B', 'SPHERE_A', 'CUBE_A', 'RUGBY_B', 'BALL_B', 'RUGBY_A', 'CUBOID_A','CUBE_C' ,'BALL_C', 'BALL_A', 'SPHERE_B', 'SPHERE_C' ,'CUBOID_B']
label_encoder = preprocessing.LabelEncoder()
z=label_encoder.fit_transform(rop)




parser = argparse.ArgumentParser(description='Add the files')
parser.add_argument('--input_data',type=str, help='location of input file')
parser.add_argument('--output',type=str, help='location of output file')
parser.add_argument('--model_path',type=str, help='location of saved model')
args = parser.parse_args()





model= torch.load( args.model_path)
inputs=pd.read_csv(args.input_data)
lens=len(inputs)
inp=inputs.iloc[:,0:33].values
scaler = preprocessing.MinMaxScaler()
x=scaler.fit_transform(inp)
val=torch.tensor(x,dtype=torch.float32)
val_data=DataLoader(val,batch_size=lens,shuffle=False)




model=model.to('cpu')
for i in val_data:
    outputs = model(i)
    _, predicted = torch.max(outputs.data, 1)
op=label_encoder.inverse_transform(predicted)





DF = pd.DataFrame(op,columns=['Object_Held'])
  
DF.to_csv(args.output,index=False)

