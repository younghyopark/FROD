import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import calculate_log as callog
import torch
import argparse
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument("--outf", default='extracted_features')
parser.add_argument("--backbone_name",'-bn', required=True)
parser.add_argument("--dataset",required=True)
parser.add_argument("--out_dataset", required=True)
parser.add_argument("--fet",default='_mean')
parser.add_argument("--fet2",default='_mean')
parser.add_argument("--prefix",required=True)
parser.add_argument("--ae_type",required=True)

parser.add_argument("--epoch",type=int,required=True)


opt = parser.parse_args()
    
ood_dataset=[opt.out_dataset,'lsun_resize','imagenet_resize'] 
#augs = ['','_cjitter', '_gray', '_hflip', '_vflip']
augs = ['']       
out_dataset=ood_dataset
num_out_datasets = len(ood_dataset)

ind_dataset=opt.dataset
experiment = opt.backbone_name
ae_type = opt.ae_type
prefix = opt.prefix #confidence
layer_num=9
epoch=opt.epoch


for idx, aug in enumerate(augs):
    if idx == 0:
        ind=[]
        ind_train=[]
        ood=dict()
        for i in range(layer_num):
            ood[i]=[]
            print(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_in_{}_epoch_{}{}{}.txt'.format(prefix, i,ind_dataset,epoch, opt.fet, aug)))
            ind.append(np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_in_{}_epoch_{}{}{}.txt'.format(prefix, i,ind_dataset,epoch, opt.fet, aug))))
            ind_train.append(np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_in_{}_epoch_{}{}_train{}.txt'.format(prefix, i,ind_dataset,epoch, opt.fet, aug))))
            for j in range(len(ood_dataset)):
                ood[i].append(np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_out_{}_epoch_{}{}_model1{}.txt'.format(prefix, i,ood_dataset[j],epoch, opt.fet, aug))))
                print(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_out_{}_epoch_{}{}_model1{}.txt'.format(prefix, i,ood_dataset[j],epoch, opt.fet, aug)))
    else:
        for i in range(layer_num):
            ind[i] += np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_in_{}_epoch_{}{}{}.txt'.format(prefix, i,ind_dataset,epoch, opt.fet, aug)))
            ind_train[i] += np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_in_{}_epoch_{}{}_train{}.txt'.format(prefix, i,ind_dataset,epoch, opt.fet, aug)))
            for j in range(len(ood_dataset)):
                ood[i][j] += np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_out_{}_epoch_{}{}_model1{}.txt'.format(prefix, i,ood_dataset[j],epoch, opt.fet, aug)))
                print(ind[i].shape,ood[i][j].shape)
ind_scaled=[]
ood_scaled=dict()
for j in range(len(ood_dataset)):
    ood_scaled[j]=[]

for i in range(layer_num):
    scaler=StandardScaler()
    scaler.fit(ind_train[i].reshape(-1,1))
    ind_scaled.append(scaler.transform(ind[i].reshape(-1,1)).reshape(-1))
    for j in range(len(ood_dataset)):
        ood_scaled[j].append(scaler.transform(ood[i][j].reshape(-1,1)).reshape(-1))
        
        
ind_scaled_max=np.min(ind_scaled,0)
ood_scaled_max=[]
for j in range(len(ood_dataset)):
    ood_scaled_max.append(np.min(ood_scaled[j],0))
        
ood_index= 0
print(ood_dataset[ood_index])
rst,_,_ = callog.metric(ind_scaled_max,ood_scaled_max[ood_index])
# print(rst)
print("{:.2f} / {:.2f} / {:.2f}".format(100*rst['TMP']['TNR'],100*rst['TMP']['AUROC'],100*rst['TMP']['DTACC']))

ood_index= 1
print(ood_dataset[ood_index])
rst,_,_ = callog.metric(ind_scaled_max,ood_scaled_max[ood_index])
# print(rst)
print("{:.2f} / {:.2f} / {:.2f}".format(100*rst['TMP']['TNR'],100*rst['TMP']['AUROC'],100*rst['TMP']['DTACC']))


ood_index= 2
print(ood_dataset[ood_index])
rst,_,_ = callog.metric(ind_scaled_max,ood_scaled_max[ood_index])
# print(rst)
print("{:.2f} / {:.2f} / {:.2f}".format(100*rst['TMP']['TNR'],100*rst['TMP']['AUROC'],100*rst['TMP']['DTACC']))


