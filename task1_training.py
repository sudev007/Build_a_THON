


from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import os
import torch
import pytorch_tabnet
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from sklearn import preprocessing
import argparse




parser = argparse.ArgumentParser(description='Add the files')
parser.add_argument('--input_data',type=str, help='location of input directory')
parser.add_argument('--model_path',type=str, help=' path for saving the model')
args = parser.parse_args()



print("Preprocessing Data")

file=os.listdir(args.input_data)
data=pd.read_csv(args.input_data+'/'+file[0])
for i in file:
    merge=pd.read_csv(args.input_data+'/'+i)
    data=data.append(merge,ignore_index=True)





label_encoder = preprocessing.LabelEncoder()
data["Size"]= label_encoder.fit_transform(data["Size"])
min_max_scaler = MinMaxScaler()
x = data.iloc[:,1:35].values
x = min_max_scaler.fit_transform(x)
y = data.iloc[:,35:37].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16, random_state=42)
model= TabNetMultiTaskClassifier()






model.fit(
    x_train,y_train,
    eval_set=[(x_train, y_train), (x_test, y_test)],
    eval_name=['train', 'valid'],
    eval_metric=['accuracy'],
    max_epochs=100,batch_size=2048
)     


saved_filepath = model.save_model(args.model_path+"/model_checkpoint_task1")



