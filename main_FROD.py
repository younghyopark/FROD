import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import calculate_log as callog
import argparse
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument("--backbone_name",'-bn', type=str, required=True)
parser.add_argument("--ae_type",'-ae', type=str, required=True)
parser.add_argument("--prefix",'-pr', type=str, default='confidence')
parser.add_argument("--dataset", type=str, default='cifar10')
opt = parser.parse_args()

ind_dataset=opt.dataset
ood_dataset=['svhn','imagenet_resize','lsun_resize','imagenet_fix','lsun_fix','place365','uniform_noise','gaussian_noise','dtd']
experiment = opt.backbone_name
ae_type = opt.ae_type
prefix = opt.prefix #confidence
layer_num=9

epoch=500
ind=[]
ind_train=[]
ood=dict()
for i in range(layer_num):
    ood[i]=[]
    ind.append(np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_in_{}_epoch_{}.txt'.format(prefix, i,ind_dataset,epoch))))
    ind_train.append(np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_in_{}_epoch_{}_train.txt'.format(prefix, i,ind_dataset,epoch))))
    for j in range(len(ood_dataset)):
        ood[i].append(np.loadtxt(os.path.join('trained_autoencoders',ae_type,experiment,'{}_layer_{}_out_{}_epoch_{}_model1.txt'.format(prefix, i,ood_dataset[j],epoch))))

from sklearn.preprocessing import StandardScaler
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


for ood_index, _ in enumerate(ood_dataset):
    f,axs=plt.subplots(1,layer_num, figsize=(60,5))
    results=dict()
    for layer in trange(layer_num,desc='plotting_original_histograms'):
        results[layer],_,_ = callog.metric(ind[layer],ood[layer][ood_index])
        sns.histplot(ax=axs[layer],data=-ind[layer],color='blue',stat='density')
        sns.histplot(ax=axs[layer],data=-ood[layer][ood_index],color='red',stat='density')
        axs[layer].legend(['In-Distribution = {}'.format(ind_dataset),'Out-of-distribution = {}'.format(ood_dataset[ood_index])])
        axs[layer].set_title('Layer '+str(layer))
        axs[layer].set_xlabel('Reconstruction Error per layer')

        rst = results[layer]['TMP']
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        metric = 'TNR = {:.5f}\nAUROC = {:.5f}\nDTACC = {:.5f}\nAUIN = {:.5f}\nAUOUT = {:.5f}'.format(100*rst['TNR'],100*rst['AUROC'],100*rst['DTACC'],100*rst['AUIN'],100*rst['AUOUT'])
        axs[layer].text(np.max((np.max(-ind[layer]),np.max(-ood[layer][ood_index]))),0, metric, fontsize=8,horizontalalignment='right', verticalalignment='bottom', bbox=props)
        f.savefig((os.path.join('trained_autoencoders',ae_type,experiment,'original_layerwise_{}.png'.format(ood_dataset[ood_index]))))

    f,axs=plt.subplots(1,layer_num, figsize=(60,5))
    results=dict()
    for layer in trange(layer_num,desc='plotting_scaled_histograms'):
        results[layer],_,_ = callog.metric(ind_scaled[layer],ood_scaled[ood_index][layer])
        sns.histplot(ax=axs[layer],data=-ind_scaled[layer],color='blue',stat='density')
        sns.histplot(ax=axs[layer],data=-ood_scaled[ood_index][layer],color='red',stat='density')
        axs[layer].legend(['In-Distribution = {}'.format(ind_dataset),'Out-of-distribution = {}'.format(ood_dataset[ood_index])])
        axs[layer].set_title('Layer '+str(layer))
        axs[layer].set_xlabel('Reconstruction Error per layer')

        rst = results[layer]['TMP']
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        metric = 'TNR = {:.5f}\nAUROC = {:.5f}\nDTACC = {:.5f}\nAUIN = {:.5f}\nAUOUT = {:.5f}'.format(100*rst['TNR'],100*rst['AUROC'],100*rst['DTACC'],100*rst['AUIN'],100*rst['AUOUT'])
        axs[layer].text(np.max((np.max(-ind_scaled[layer]),np.max(-ood_scaled[ood_index][layer]))),0, metric, fontsize=8,horizontalalignment='right', verticalalignment='bottom', bbox=props)
        f.savefig((os.path.join('trained_autoencoders',ae_type,experiment,'scaled_layerwise_{}.png'.format(ood_dataset[ood_index]))))

    plt.figure()
    results_max,_,_ = callog.metric(ind_scaled_max,ood_scaled_max[ood_index])
    sns.histplot(data=-ind_scaled_max,color='blue',stat='density')
    sns.histplot(data=-ood_scaled_max[ood_index],color='red',stat='density')
    plt.legend(['In-Distribution = {}'.format(ind_dataset),'Out-of-distribution = {}'.format(ood_dataset[ood_index])])
    plt.title('Max-Anomaly')
    plt.xlabel('Reconstruction Error per layer')

    print(ood_dataset[ood_index])
    rst = results_max['TMP']
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    metric = 'TNR = {:.5f}\nAUROC = {:.5f}\nDTACC = {:.5f}\nAUIN = {:.5f}\nAUOUT = {:.5f}'.format(100*rst['TNR'],100*rst['AUROC'],100*rst['DTACC'],100*rst['AUIN'],100*rst['AUOUT'])
    print("{:.2f} / {:.2f} / {:.2f}".format(100*rst['TNR'],100*rst['AUROC'],100*rst['DTACC']))
    plt.text(np.max((np.max(-ind_scaled_max),np.max(-ood_scaled_max[ood_index]))),0, metric, fontsize=8,horizontalalignment='right', verticalalignment='bottom', bbox=props)
    plt.savefig((os.path.join('trained_autoencoders',ae_type,experiment,'maxanomaly_{}.png'.format(ood_dataset[ood_index]))))
    plt.close()
