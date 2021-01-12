import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter


import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import MNIST
import shutil


from plotly.offline import plot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
#python ./autoencoder_training/vanillaAE_resnet18.py --backbone_name resnet18_vanilla_simclr_svhn --gpu 6 --dataset svhn --out_dataset cifar10 --out_dataset9 cifar100
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument('--backbone_name','-bn', required=True, help='')
parser.add_argument('--gpu', type=int, required=True, help='gpu index')
parser.add_argument('--ckpt_epoch',type=int,default=50)
parser.add_argument('--test_epoch',type=int,default=20)
parser.add_argument("--out_target", type=int, default=100)
parser.add_argument("--dataset", default = 'cifar10')
parser.add_argument("--out_dataset",default='svhn')
parser.add_argument("--out_dataset2",default='imagenet_resize')
parser.add_argument("--out_dataset3",default='lsun_resize')
parser.add_argument("--out_dataset4",default='imagenet_fix')
parser.add_argument("--out_dataset5",default='lsun_fix')
parser.add_argument("--out_dataset6",default='None')
parser.add_argument("--out_dataset7",default='None')
parser.add_argument("--out_dataset8",default='None')
parser.add_argument("--out_dataset9",default='None')
parser.add_argument('--outf',default='extracted_features')
parser.add_argument('--resume',type=int, default=0)
parser.add_argument('--feature_extraction_type','-fet',type=str, default='mean')


parser.add_argument('--moco_version','-v',type=int, default=0)


opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

device = torch.device('cuda')
torch.cuda.set_device(opt.gpu)

writer = SummaryWriter(logdir=os.path.join('trained_autoencoders','deep_svdd',opt.backbone_name))

out_dataset = []
num_out_datasets=0

if opt.dataset in ['cifar100','cifar10','svhn']:
    out_dataset_temp = [opt.out_dataset,opt.out_dataset2,opt.out_dataset3,opt.out_dataset4,opt.out_dataset5,opt.out_dataset6,opt.out_dataset7,opt.out_dataset8,opt.out_dataset9]

    for out in out_dataset_temp:
        if out!='None':
            out_dataset.append(out)
            num_out_datasets+=1
else:
    num_out_datasets = 1
    out_dataset = ['MNIST']
layer_num=9
if opt.moco_version==1:
    layer_num=10
elif opt.moco_version==2:
    layer_num=14
    
layer_num = 12
print('layer num', layer_num)

train_ind_feature=dict()
test_ind_feature=dict()
test_ood_feature=dict()
num_ood=dict()
for i in range(layer_num):
    test_ood_feature[i]=[]
    num_ood[i]=[]
    train_ind_feature[i]=np.load(os.path.join(opt.outf,opt.backbone_name,'Features_from_layer_'+str(i)+'_'+opt.dataset+'_'+opt.feature_extraction_type+'_train_ind.npy'))
    test_ind_feature[i]=np.load(os.path.join(opt.outf,opt.backbone_name,'Features_from_layer_'+str(i)+'_'+opt.dataset+'_'+opt.feature_extraction_type+'_test_ind.npy'))
    print(num_out_datasets)
    for j in range(num_out_datasets):
        test_ood_feature[i].append(np.load(os.path.join(opt.outf,opt.backbone_name,'Features_from_layer_'+str(i)+'_'+out_dataset[j]+'_'+opt.feature_extraction_type+'_test_ood.npy')))
        num_ood[i].append(test_ood_feature[i][j].shape[0])
train_data_ind = train_ind_feature
test_data_ind = test_ind_feature
test_data_ood = test_ood_feature
for i in range(layer_num):
    print(train_data_ind[i].shape)
    
        
class SVDD(nn.Module):

    def __init__(self,x_dim, h_dim1, h_dim2,h_dim3,h_dim4,h_dim5,h_dim6,R,c):
        super(SVDD, self).__init__()
        self.c = nn.Parameter(torch.tensor(c), requires_grad=False)
        self.R = nn.Parameter(torch.tensor(R),requires_grad=False)


        self.h_dim6=h_dim6
        self.h_dim5=h_dim5
        self.h_dim4=h_dim4
        self.fc1 = nn.Linear(x_dim, h_dim1,bias=False)
        self.bn1 = nn.BatchNorm1d(h_dim1, affine=False)
        self.fc2 = nn.Linear(h_dim1, h_dim2,bias=False)
        self.bn2 = nn.BatchNorm1d(h_dim2, affine=False)
        self.fc3 = nn.Linear(h_dim2, h_dim3,bias=False)
        self.bn3 = nn.BatchNorm1d(h_dim3, affine=False)

        if h_dim4>0:
            self.fc4 = nn.Linear(h_dim3,h_dim4,bias=False)
            self.bn4 = nn.BatchNorm1d(h_dim4, affine=False)
        if h_dim5 >0:
            self.fc5 = nn.Linear(h_dim4,h_dim5,bias=False)
            self.bn5 = nn.BatchNorm1d(h_dim5, affine=False)

        if h_dim6 >0:
            self.fc6 = nn.Linear(h_dim5,h_dim6,bias=False)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn2(x))
        if self.h_dim6 >0:
            x = self.fc3(x)
            x = F.leaky_relu(self.bn3(x))
            x = self.fc4(x)
            x = F.leaky_relu(self.bn4(x))
            x = self.fc5(x)
            x = F.leaky_relu(self.bn5(x))
            h = self.fc6(h)
        elif self.h_dim5 >0:
            x = self.fc3(x)
            x = F.leaky_relu(self.bn3(x))
            x = self.fc4(x)
            x = F.leaky_relu(self.bn4(x))
            x = self.fc5(x)
        elif self.h_dim4 >0:
            x = self.fc3(x)
            x = F.leaky_relu(self.bn3(x))
            x = self.fc4(x)
        else:
            x = self.fc3(x)
            
        dist = torch.sum((x - self.c) ** 2, dim=1)

        return dist
    
    
    def get_feature(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn2(x))
        if self.h_dim6 >0:
            x = self.fc3(x)
            x = F.leaky_relu(self.bn3(x))
            x = self.fc4(x)
            x = F.leaky_relu(self.bn4(x))
            x = self.fc5(x)
            x = F.leaky_relu(self.bn5(x))
            h = self.fc6(h)
        elif self.h_dim5 >0:
            x = self.fc3(x)
            x = F.leaky_relu(self.bn3(x))
            x = self.fc4(x)
            x = F.leaky_relu(self.bn4(x))
            x = self.fc5(x)
        elif self.h_dim4 >0:
            x = self.fc3(x)
            x = F.leaky_relu(self.bn3(x))
            x = self.fc4(x)
        else:
            x = self.fc3(x)
            
        return x

    def init_center_c(self, train_loader, eps=0.01):
        n_samples = 0
#         self.c = torch.zeros(self.rep_dim, device='cuda')
        with torch.no_grad():
            for data in train_loader:
                data = data.cuda()
                feature = self.get_feature(data)
                n_samples += data.size(0)
                self.c += torch.sum(feature, dim=0)
        
        self.c /= n_samples
        self.c[(abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(abs(self.c) < eps) & (self.c > 0)] = eps
        
        return self.c


models=dict()
models[0] = SVDD(64, 32, 16, 16,0,0,0,R=0.0,c=torch.zeros(16))
models[1] = SVDD(64, 32, 16, 16,0,0,0,R=0.0,c=torch.zeros(16))
models[2] = SVDD(64, 32, 16, 16,0,0,0,R=0.0,c=torch.zeros(16))

models[3] = SVDD(128, 64, 32, 32,0,0,0,R=0.0,c=torch.zeros(32))
models[4] = SVDD(128, 64, 32, 32,0,0,0,R=0.0,c=torch.zeros(32))

models[5] = SVDD(256, 128, 64, 64, 0, 0,0,R=0.0,c=torch.zeros(64))
models[6] = SVDD(256, 128, 64, 64, 0, 0,0,R=0.0,c=torch.zeros(64))

models[7] = SVDD(512,256,128,128,0,0,0,R=0.0,c=torch.zeros(128))
models[8] = SVDD(512,256,128,128,0,0,0,R=0.0,c=torch.zeros(128))
models[9] = SVDD(512,256,128,128,0,0,0,R=0.0,c=torch.zeros(128))

models[10] = SVDD(2048,512,256,256,0,0,0,R=0.0,c=torch.zeros(256))
models[11] = SVDD(128, 64, 32, 32,0,0,0,R=0.0,c=torch.zeros(32))


device = torch.device('cuda')
torch.cuda.set_device(opt.gpu)
optimizer=dict()
schedular=dict()
for i in range(layer_num):
    optimizer[i] = torch.optim.Adam(models[i].parameters(), opt.lr)
    schedular[i] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[i], T_max=opt.n_epochs, eta_min=0, last_epoch=-1)

train_ind_loader=dict()
test_ind_loader=dict()
test_ood_loader=dict()
for i in range(layer_num):
    train_ind_loader[i] = torch.utils.data.DataLoader(torch.Tensor(train_data_ind[i]), batch_size=opt.batch_size, shuffle=True)
    test_ind_loader[i] = torch.utils.data.DataLoader(torch.Tensor(test_data_ind[i]), batch_size=opt.batch_size, shuffle=False)
    test_ood_loader[i]=[]
    for j in range(num_out_datasets):
        test_ood_loader[i].append(torch.utils.data.DataLoader(torch.Tensor(test_data_ood[i][j]), batch_size=opt.batch_size, shuffle=False))
    models[i].to(device)
    
    if opt.resume==0:

        models[i].train()

        models[i].c=models[i].init_center_c(train_ind_loader[i])
        print('Init center complete:', models[i].c)

        for epoch in range(1, opt.n_epochs + 1):
            avg_loss = 0
            step = 0
            loss_epoch = 0.0
            n_batches = 0

            for data in train_ind_loader[i]:
                step += 1
        #         data = torch.cat((ind,ood),dim=0)
                data = data.cuda()
                outputs = models[i].get_feature(data)
                optimizer[i].zero_grad()

                dist= torch.sum((outputs - models[i].c) ** 2, dim=1)

                loss = torch.mean(dist)
  
                loss.backward()
 
                loss_epoch += loss.item()
                n_batches += 1

            print('Epoch [{}/{}] => radius: {:.5f}, loss: {:.5f}'.format(epoch, opt.n_epochs, models[i].R, loss_epoch/n_batches))

            if epoch % opt.ckpt_epoch == 0:
                model_state = models[i].state_dict()
                #print(model_state)
                ckpt_name = 'layer_{}_epoch_{}'.format(i,epoch)
                ckpt_path = os.path.join('trained_autoencoders','deep_svdd',opt.backbone_name,ckpt_name + ".pth")
                torch.save(model_state, ckpt_path)

if opt.resume==1:
    epoch = 500
    for j in range(layer_num):
        ckpt_name = 'layer_{}_epoch_500'.format(j)
        tm = torch.load(os.path.join('trained_autoencoders','deep_svdd',opt.backbone_name,ckpt_name + ".pth"))
        print('model {} loaded'.format(j))

print('=== reconstruction error calculation on test data ===')
for j in range(layer_num):
    models[j].eval()
    print('layer {}'.format(j))
    rc_error_ind = []
    for i, data in enumerate(tqdm(train_ind_loader[j])):
        data = data.cuda()
#             print(i)
        recon_error = models[j](data)
        rc_error_ind.append(recon_error)
    rc_error_ind_total = torch.cat(rc_error_ind,0)   
    rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
    ind_score = -rc_error_ind_total_np
    l0 = open('./trained_autoencoders/deep_svdd/'+opt.backbone_name+'/distance_layer_{}_in_{}_epoch_{}_{}_train.txt'.format(j,opt.dataset,epoch, opt.feature_extraction_type), 'w')
    for i in range(ind_score.shape[0]):
        l0.write("{}\n".format(ind_score[i]))
    l0.close()

    rc_error_ind = []
    for i, data in enumerate(tqdm(test_ind_loader[j])):
        data = data.cuda()
#             print(i)
        recon_error = models[j](data)
        rc_error_ind.append(recon_error)
    rc_error_ind_total = torch.cat(rc_error_ind,0)   
    rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
    ind_score = -rc_error_ind_total_np
    l1 = open('./trained_autoencoders/deep_svdd/'+opt.backbone_name+'/distance_layer_{}_in_{}_epoch_{}_{}.txt'.format(j,opt.dataset,epoch, opt.feature_extraction_type), 'w')
    for i in range(ind_score.shape[0]):
        l1.write("{}\n".format(ind_score[i]))
    l1.close()

    for out_n in range(num_out_datasets):
        rc_error_ood = []
        for i, data in enumerate(tqdm(test_ood_loader[j][out_n])):
            data = data.cuda()
#             print(i)
            recon_error = models[j](data)
            rc_error_ood.append(recon_error)
        rc_error_ood_total = torch.cat(rc_error_ood,0)   
        rc_error_ood_total_np = rc_error_ood_total.detach().cpu().numpy()        

        ood_score = -rc_error_ood_total_np
        l2 = open('./trained_autoencoders/deep_svdd/'+opt.backbone_name+'/distance_layer_{}_out_{}_epoch_{}_{}_model1.txt'.format(j,out_dataset[out_n],epoch, opt.feature_extraction_type), 'w')
        for i in range(ood_score.shape[0]):
            l2.write("{}\n".format(ood_score[i]))
        l2.close()

