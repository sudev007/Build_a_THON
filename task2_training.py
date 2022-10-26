
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
splits=KFold(n_splits=10,shuffle=True,random_state=42)
batch_size=1024
num_epochs=250
foldperf={}
criterion = nn.CrossEntropyLoss()





parser = argparse.ArgumentParser(description='Add the files')
parser.add_argument('--input_data',type=str, help='location of input file')
parser.add_argument('--model_path',type=str, help=' path for saving the model')
args = parser.parse_args()





class MyDataset(Dataset):
 
  def __init__(self,file_name):
    price_df=pd.read_csv(file_name)
    scaler = preprocessing.MinMaxScaler()
    x=price_df.iloc[:,0:33].values
    y=price_df.iloc[:,33]
    x=scaler.fit_transform(x)
    label_encoder = preprocessing.LabelEncoder()
    y= label_encoder.fit_transform(y)
    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

loader=MyDataset(args.input_data)






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






def train_epoch(model,device,dataloader,loss_fn,optimizer):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:

        inputs,labels = images.to(device),labels.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss,train_correct
  
def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:

        inputs,labels = images.to(device),labels.type(torch.LongTensor).to(device)
        output = model(inputs)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*inputs.size(0)
        scores, predictions = torch.max(output.data,1)
        val_correct+=(predictions == labels).sum().item()

    return valid_loss,val_correct






for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(loader)))):

    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(loader, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(loader, batch_size=batch_size, sampler=test_sampler)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Net()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    for epoch in range(num_epochs):
        train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
        test_loss, test_correct=valid_epoch(model,device,test_loader,criterion)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                             num_epochs,
                                                                                                             train_loss,
                                                                                                             test_loss,
                                                                                                             train_acc,
                                                                                                             test_acc))
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    foldperf['fold{}'.format(fold+1)] = history  





torch.save(model, args.model_path+"/model_checkpoint_task2")





