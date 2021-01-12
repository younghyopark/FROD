import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import calculate_log as callog
import torch
import argparse
from sklearn.preprocessing import StandardScaler
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument("--outf", default='extracted_features')
parser.add_argument("--backbone_name",'-bn', required=True)
parser.add_argument("--dataset",required=True)
parser.add_argument("--out_dataset", required=True)
parser.add_argument("--fet",default='')
parser.add_argument("--fet2",default='_mean')
parser.add_argument("--prefix",required=True)
parser.add_argument("--ae_type",required=True)

parser.add_argument("--epoch",type=int,required=True)
parser.add_argument("--coef",type=float,required=True)



opt = parser.parse_args()

ood_dataset=[opt.out_dataset,'lsun_resize','imagenet_resize','lsun_fix','imagenet_fix'] 
out_dataset=ood_dataset
num_out_datasets = len(ood_dataset)

ind_dataset=opt.dataset
experiment = opt.backbone_name
ae_type = opt.ae_type
prefix = opt.prefix #confidence

layer_num=9
epoch=opt.epoch
ind=[]
ind_train=[]
ood=dict()
for i in trange(layer_num):
    ood[i]=[]
    ind.append(np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_in_{}_epoch_{}{}.txt'.format(prefix, i,ind_dataset,epoch, opt.fet))))
    ind_train.append(np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_in_{}_epoch_{}{}_train.txt'.format(prefix, i,ind_dataset,epoch, opt.fet))))
    for j in range(len(ood_dataset)):
        ood[i].append(np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_out_{}_epoch_{}{}_model1.txt'.format(prefix, i,ood_dataset[j],epoch, opt.fet))))
        
        
ind_scaled=[]
ood_scaled=dict()
for j in range(len(ood_dataset)):
    ood_scaled[j]=[]

for i in trange(layer_num):
    scaler=StandardScaler()
    scaler.fit(ind_train[i].reshape(-1,1))
    ind_scaled.append(scaler.transform(ind[i].reshape(-1,1)).reshape(-1))
    for j in range(len(ood_dataset)):
        ood_scaled[j].append(scaler.transform(ood[i][j].reshape(-1,1)).reshape(-1))

ind_scaled_max=np.min(ind_scaled,0)
ood_scaled_max=[]
for j in range(len(ood_dataset)):
    ood_scaled_max.append(np.min(ood_scaled[j],0))


train_ind_feature=dict()
test_ind_feature=dict()
test_ood_feature=dict()
num_ood=dict()

for i in trange(layer_num+3):
    test_ood_feature[i]=[]
    num_ood[i]=[]
    train_ind_feature[i]=np.load(os.path.join(opt.outf,opt.backbone_name,'Features_from_layer_'+str(i)+'_'+opt.dataset+opt.fet2+'_train_ind.npy'))
    test_ind_feature[i]=np.load(os.path.join(opt.outf,opt.backbone_name,'Features_from_layer_'+str(i)+'_'+opt.dataset+opt.fet2+'_test_ind.npy'))
    for j in range(num_out_datasets):
        test_ood_feature[i].append(np.load(os.path.join(opt.outf,opt.backbone_name,'Features_from_layer_'+str(i)+'_'+out_dataset[j]+opt.fet2+'_test_ood.npy')))
        num_ood[i].append(test_ood_feature[i][j].shape[0])
train_data_ind = train_ind_feature
test_data_ind = test_ind_feature
test_data_ood = test_ood_feature
# for i in range(layer_num+3):
#     print(train_data_ind[i].shape)
    
train_ind_norm=dict()
test_ind_norm=dict()
test_ood_norm=dict()

for i in range(layer_num+3):
    layer=i
    test_ood_norm[i]=[]
    num_ood[i]=[]
    train_ind_norm[i]=np.linalg.norm(train_ind_feature[layer],axis=1)
    test_ind_norm[i]=np.linalg.norm(test_ind_feature[layer],axis=1)
    for j in range(num_out_datasets):
        test_ood_norm[i].append(np.linalg.norm(test_ood_feature[layer][j],axis=1))

        
        
for cf1 in [0,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0,100.0]:
    print('coefficient = ',cf1)
    ood_final=[]
#     cf1 =opt.coef
    cf2= 1
    ind_final=(-ind_scaled_max)-(test_ind_norm[layer_num+2])*cf1
    # ind_final=ind_scaled_max/(test_ind_norm[layer_num-1])


    for j in range(len(ood_dataset)):
        ood_final.append(
    #         -(ood_nn_scaled_max[j])+(test_ood_norm[layer_num-1][j])+ood_scaled_max[j]
            1*(-ood_scaled_max[j])-(test_ood_norm[layer_num+2][j])*cf1
        )    


    ood_index= 0
    print(ood_dataset[ood_index])
    rst,_,_ = callog.metric(-ind_final,-ood_final[ood_index])
    # print(rst)
    print("{:.2f} / {:.2f} / {:.2f}".format(100*rst['TMP']['TNR'],100*rst['TMP']['AUROC'],100*rst['TMP']['DTACC']))

    ood_index= 1
    print(ood_dataset[ood_index])
    rst,_,_ = callog.metric(-ind_final,-ood_final[ood_index])
    # print(rst)
    print("{:.2f} / {:.2f} / {:.2f}".format(100*rst['TMP']['TNR'],100*rst['TMP']['AUROC'],100*rst['TMP']['DTACC']))


    ood_index= 2
    print(ood_dataset[ood_index])
    rst,_,_ = callog.metric(-ind_final,-ood_final[ood_index])
    # print(rst)
    print("{:.2f} / {:.2f} / {:.2f}".format(100*rst['TMP']['TNR'],100*rst['TMP']['AUROC'],100*rst['TMP']['DTACC']))

    
    ood_index= 3
    print(ood_dataset[ood_index])
    rst,_,_ = callog.metric(-ind_final,-ood_final[ood_index])
    # print(rst)
    print("{:.2f} / {:.2f} / {:.2f}".format(100*rst['TMP']['TNR'],100*rst['TMP']['AUROC'],100*rst['TMP']['DTACC']))

    ood_index= 4
    print(ood_dataset[ood_index])
    rst,_,_ = callog.metric(-ind_final,-ood_final[ood_index])
    # print(rst)
    print("{:.2f} / {:.2f} / {:.2f}".format(100*rst['TMP']['TNR'],100*rst['TMP']['AUROC'],100*rst['TMP']['DTACC']))

    print('')

