


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
parser.add_argument('--input_data',type=str, help='location of input folder')
parser.add_argument('--output',type=str, help='location of output file')
parser.add_argument('--model_path',type=str, help='location of saved model')
args = parser.parse_args()




model= TabNetMultiTaskClassifier()
model.load_model(args.model_path)



file=os.listdir(args.input_data)

for i in file:
    data=pd.read_csv(args.input_data+'/'+i)
    min_max_scaler = MinMaxScaler()
    label_encoder = preprocessing.LabelEncoder()
    data["Size"]= label_encoder.fit_transform(data["Size"])
    test = data.iloc[:,1:35].values
    test = min_max_scaler.fit_transform(test)
    preds=model.predict(test)
    final=pd.DataFrame(list(zip(preds[0], preds[1])),columns =['Slip', 'Crumple'])
    final.to_csv(args.output+'/'+i,index=False)

