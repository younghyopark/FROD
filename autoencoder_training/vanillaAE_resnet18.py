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

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument('--backbone_name', required=True, help='')
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
parser.add_argument("--out_dataset6",default='dtd')
parser.add_argument("--out_dataset7",default='place365')
parser.add_argument("--out_dataset8",default='gaussian_noise')
parser.add_argument("--out_dataset9",default='uniform_noise')
parser.add_argument('--outf',default='extracted_features')
parser.add_argument('--resume',type=int, default=0)

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

device = torch.device('cuda')
torch.cuda.set_device(opt.gpu)

writer = SummaryWriter(logdir=os.path.join('trained_autoencoders','vanilla_AE',opt.backbone_name))

out_dataset = []
num_out_datasets=0

if opt.dataset in ['cifar10','svhn']:
    out_dataset_temp = [opt.out_dataset,opt.out_dataset2,opt.out_dataset3,opt.out_dataset4,opt.out_dataset5,opt.out_dataset6,opt.out_dataset7,opt.out_dataset8,opt.out_dataset9]

    for out in out_dataset_temp:
        if out!='None':
            out_dataset.append(out)
            num_out_datasets+=1
else:
    num_out_datasets = 1
    out_dataset = ['MNIST']

train_ind_feature=dict()
test_ind_feature=dict()
test_ood_feature=dict()
num_ood=dict()
for i in range(9):
    test_ood_feature[i]=[]
    num_ood[i]=[]
    train_ind_feature[i]=np.load(os.path.join(opt.outf,opt.backbone_name,'Features_from_layer_'+str(i)+'_'+opt.dataset+'_'+'original'+'_train_ind.npy'))
    test_ind_feature[i]=np.load(os.path.join(opt.outf,opt.backbone_name,'Features_from_layer_'+str(i)+'_'+opt.dataset+'_'+'original'+'_test_ind.npy'))
    print(num_out_datasets)
    for j in range(num_out_datasets):
        test_ood_feature[i].append(np.load(os.path.join(opt.outf,opt.backbone_name,'Features_from_layer_'+str(i)+'_'+out_dataset[j]+'_'+'original'+'_test_ood.npy')))
        num_ood[i].append(test_ood_feature[i][j].shape[0])
train_data_ind = train_ind_feature
test_data_ind = test_ind_feature
test_data_ood = test_ood_feature
for i in range(9):
    print(train_data_ind[i].shape)

class AE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3,h_dim4,h_dim5,h_dim6):
        super(AE, self).__init__()
        self.x_dim = x_dim
        # encoder part
        self.encoder = Encoder(x_dim, h_dim1, h_dim2,  h_dim3,h_dim4,h_dim5, h_dim6)
        # decoder part
        self.decoder = Generator(x_dim, h_dim1, h_dim2, h_dim3,h_dim4,h_dim5,h_dim6)
    
    def recon_error(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return torch.norm((x_recon - x), dim=1)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    
class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,h_dim5,h_dim6):
        super(Encoder, self).__init__()
        self.h_dim6=h_dim6
        self.h_dim5=h_dim5
        self.h_dim4=h_dim4
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim3)
        if h_dim4>0:
            self.fc4 = nn.Linear(h_dim3,h_dim4)
        if h_dim5 >0:
            self.fc5 = nn.Linear(h_dim4,h_dim5)
        if h_dim6 >0:
            self.fc6 = nn.Linear(h_dim5,h_dim6)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        if self.h_dim6 >0:
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc5(h))
            h = self.fc6(h)
        elif self.h_dim5 >0:
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc4(h))
            h = self.fc5(h)
        elif self.h_dim4 >0:
            h = F.relu(self.fc3(h))
            h = self.fc4(h)
        else:
            h = self.fc3(h)
        return h
    
    
class Generator(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,h_dim5, h_dim6):
        super(Generator, self).__init__()
        self.h_dim6=h_dim6
        self.h_dim5=h_dim5
        self.h_dim4=h_dim4
        if h_dim6 >0:
            self.fc6 = nn.Linear(h_dim6,h_dim5)
        if h_dim5 >0:
            self.fc5 = nn.Linear(h_dim5,h_dim4)
        if h_dim4 >0:
            self.fc4 = nn.Linear(h_dim4,h_dim3)
        self.fc3 = nn.Linear(h_dim3, h_dim2)
        self.fc2 = nn.Linear(h_dim2, h_dim1)
        self.fc1 = nn.Linear(h_dim1, x_dim)
    
    def forward(self, z):
        if self.h_dim6 >0:
            h = F.relu(self.fc6(z))
            h = F.relu(self.fc5(h))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc3(h))
        elif self.h_dim5 >0:
            h = F.relu(self.fc5(z))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc3(h))
        elif self.h_dim4>0:
            h = F.relu(self.fc4(z))
            h = F.relu(self.fc3(h))
        else:
            h = F.relu(self.fc3(z))

        h = F.relu(self.fc2(h))
        return self.fc1(h)

models=dict()
models[0] = AE(64, 32, 16, 8,4,0,0)
models[1] = AE(64, 32, 16, 8,4,0,0)
models[2] = AE(64, 32, 16, 8,4,0,0)

models[3] = AE(128, 64, 32, 16,8,4,0)
models[4] = AE(128, 64, 32, 16,8,4,0)

models[5] = AE(256, 128, 64, 32, 16, 8,4)
models[6] = AE(256, 128, 64, 32, 16, 8,4)
models[7] = AE(512,256,128,64,32,8,4)
models[8] = AE(512,256,128,64,32,8,4)


optimizer=dict()
schedular=dict()
for i in range(9):
    optimizer[i] = torch.optim.Adam(models[i].parameters(), opt.lr)
    schedular[i] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[i], T_max=opt.n_epochs, eta_min=0, last_epoch=-1)
   
    
train_ind_loader=dict()
test_ind_loader=dict()
test_ood_loader=dict()
for i in range(9):
    train_ind_loader[i] = torch.utils.data.DataLoader(torch.Tensor(train_data_ind[i]), batch_size=opt.batch_size, shuffle=True)
    test_ind_loader[i] = torch.utils.data.DataLoader(torch.Tensor(test_data_ind[i]), batch_size=opt.batch_size, shuffle=False)
    test_ood_loader[i]=[]
    for j in range(num_out_datasets):
        test_ood_loader[i].append(torch.utils.data.DataLoader(torch.Tensor(test_data_ood[i][j]), batch_size=opt.batch_size, shuffle=False))
    models[i].to(device)
    models[i].train()

if opt.resume==0:
    for j in range(9):
        for epoch in range(1, opt.n_epochs+ 1):
            avg_loss = 0
            step = 0
            for i, data in enumerate(train_ind_loader[j]):
                step += 1
                data = data.cuda()
        #         print(data)
                optimizer[j].zero_grad()
                recon_error = models[j].recon_error(data)
                loss = torch.mean(recon_error)
                loss.backward()
                optimizer[j].step()
                avg_loss += loss
                if i % 100 == 0:    
                    print('Model for layer {} => Epoch [{}/{}] Batch [{}/{}]=> Loss: {:.5f}'.format(j,epoch, opt.n_epochs, i,len(train_ind_loader[j]), avg_loss / step))

            if epoch % opt.ckpt_epoch == 0:
                model_state = models[j].state_dict()
                #print(model_state)
                ckpt_name = 'layer_{}_epoch_{}'.format(j,epoch)
                ckpt_path = os.path.join('trained_autoencoders','vanilla_AE',opt.backbone_name,ckpt_name + ".pth")
                torch.save(model_state, ckpt_path)

if opt.resume==1:
    epoch = 500
    for j in range(9):
        ckpt_name = 'layer_{}'.format(j)
        tm = torch.load(os.path.join('trained_autoencoders','vanilla_AE',opt.backbone_name,ckpt_name + ".pth"))
        print('model {} loaded'.format(j))

print('=== reconstruction error calculation on test data ===')
for j in range(9):
    print('epoch {}'.format(j))
    rc_error_ind = []
    for i, data in enumerate(tqdm(train_ind_loader[j])):
        data = data.cuda()
#             print(i)
        recon_error = models[j].recon_error(data)
        rc_error_ind.append(recon_error)
    rc_error_ind_total = torch.cat(rc_error_ind,0)   
    rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
    ind_score = -rc_error_ind_total_np
    l0 = open('./trained_autoencoders/vanilla_AE/'+opt.backbone_name+'/confidence_layer_{}_in_{}_epoch_{}_train.txt'.format(j,opt.dataset,epoch), 'w')
    for i in range(ind_score.shape[0]):
        l0.write("{}\n".format(ind_score[i]))
    l0.close()

    rc_error_ind = []
    for i, data in enumerate(tqdm(test_ind_loader[j])):
        data = data.cuda()
#             print(i)
        recon_error = models[j].recon_error(data)
        rc_error_ind.append(recon_error)
    rc_error_ind_total = torch.cat(rc_error_ind,0)   
    rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
    ind_score = -rc_error_ind_total_np
    l1 = open('./trained_autoencoders/vanilla_AE/'+opt.backbone_name+'/confidence_layer_{}_in_{}_epoch_{}.txt'.format(j,opt.dataset,epoch), 'w')
    for i in range(ind_score.shape[0]):
        l1.write("{}\n".format(ind_score[i]))
    l1.close()

    for out_n in range(num_out_datasets):
        rc_error_ood = []
        for i, data in enumerate(tqdm(test_ood_loader[j][out_n])):
            data = data.cuda()
#             print(i)
            recon_error = models[j].recon_error(data)
            rc_error_ood.append(recon_error)
        rc_error_ood_total = torch.cat(rc_error_ood,0)   
        rc_error_ood_total_np = rc_error_ood_total.detach().cpu().numpy()        

        ood_score = -rc_error_ood_total_np
        l2 = open('./trained_autoencoders/vanilla_AE/'+opt.backbone_name+'/confidence_layer_{}_out_{}_epoch_{}_model1.txt'.format(j,out_dataset[out_n],epoch), 'w')
        for i in range(ood_score.shape[0]):
            l2.write("{}\n".format(ood_score[i]))
        l2.close()
