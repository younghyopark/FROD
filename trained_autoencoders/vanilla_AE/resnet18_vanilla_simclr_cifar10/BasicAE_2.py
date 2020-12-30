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
import calculate_log as callog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import MNIST
import shutil


from plotly.offline import plot
import plotly.graph_objs as go
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--h_dim1", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--h_dim2", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--h_dim3", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--h_dim4", type=int, default=0, help="number of classes for dataset")
parser.add_argument("--h_dim5", type=int, default=0, help="number of classes for dataset")
parser.add_argument("--h_dim6", type=int, default=0, help="number of classes for dataset")
parser.add_argument('--experiment', required=True, help='')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--layer',default='original')
parser.add_argument('--multiclass',default='single')
parser.add_argument('--ckpt_epoch',type=int,default=200)
parser.add_argument('--test_epoch',type=int,default=20)
parser.add_argument('--ae_name', help='')
# parser.add_argument('--division', type=int,required=True, help='')
parser.add_argument('--char',required=True, help='')
parser.add_argument("--out_target", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--dataset")
parser.add_argument("--out_dataset",default='None')
parser.add_argument("--out_dataset2",default='imagenet_resize')
parser.add_argument("--out_dataset3",default='lsun_resize')
parser.add_argument("--out_dataset4",default='imagenet_fix')
parser.add_argument("--out_dataset5",default='lsun_fix')
parser.add_argument("--out_dataset6",default='dtd')
parser.add_argument("--out_dataset7",default='place365')
parser.add_argument("--out_dataset8",default='gaussian_noise')
parser.add_argument("--out_dataset9",default='uniform_noise')
parser.add_argument('--char_feature_layer',type=int)
parser.add_argument('--outf',default='output_simclr_3_features')
parser.add_argument('--howmany',default=18)


opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

device = torch.device('cuda')
torch.cuda.set_device(opt.gpu)
# opt.ae_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_scaling'.format(opt.experiment,opt.char,opt.layer,opt.multiclass,opt.h_dim1,opt.h_dim2,opt.h_dim3,opt.h_dim4,opt.h_dim5,opt.h_dim6)


writer = SummaryWriter(logdir=os.path.join('BasicAE_layerwise_TEST',opt.experiment,opt.ae_name))
shutil.copy('BasicAE_2.py',os.path.join('BasicAE_layerwise_TEST',opt.experiment,opt.ae_name))

experiment = opt.experiment
layer =opt.layer
multi = opt.multiclass

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

if opt.char == 'MD':


    scaler = StandardScaler()
    train_data_ind=np.load(os.path.join('/HDD1','ParkYH','deep_Mahalanobis_detector',opt.outf,experiment,'Mahalanobis_in_{}_train_{}_{}.npy'.format(opt.dataset,opt.layer,opt.multiclass)))[:,:-1]
    train_data_ind=scaler.fit_transform(train_data_ind)
    
    test_data_ind=np.load(os.path.join('/HDD1','ParkYH','deep_Mahalanobis_detector',opt.outf,experiment,'Mahalanobis_in_{}_{}_{}.npy'.format(opt.dataset,opt.layer,opt.multiclass)))[:,:-1]
    test_data_ind=scaler.transform(test_data_ind)

    test_data_ood=[]
    num_ood = []
    for i,out in enumerate(out_dataset):
        test_data_ood.append(np.load(os.path.join('/HDD1','ParkYH','deep_Mahalanobis_detector',opt.outf,experiment,'Mahalanobis_out_{}_{}_{}.npy'.format(out,opt.layer,opt.multiclass)))[:,:-1])
        num_ood.append(test_data_ood[i].shape[0])
        test_data_ood[i]=scaler.transform(test_data_ood[i])

    train_ind_target=np.load(os.path.join('/HDD1','ParkYH','deep_Mahalanobis_detector',opt.outf,experiment,'Targets_in_{}_{}_{}.npy'.format(opt.dataset,opt.layer,opt.multiclass)))
    
    layer_num = train_data_ind.shape[1]
    print(layer_num)
    print(train_data_ind)
    
elif opt.char=='image':
    
    layer_num = 1024

    train_data_temp = MNIST('./',train=True).data/255
    test_data_temp = MNIST('./',train=False).data/255
    train_target_temp =  MNIST('./',train=True).targets
    test_target_temp =  MNIST('./',train=False).targets
    
    test_ind_index = [i for i in range(test_target_temp.shape[0]) if test_target_temp[i]!=opt.out_target]
    test_ood_index = [i for i in range(test_target_temp.shape[0]) if test_target_temp[i]==opt.out_target]
    train_ind_index = [i for i in range(train_target_temp.shape[0]) if train_target_temp[i]!=opt.out_target]
    print(len(train_ind_index))
    train_ood_index = [i for i in range(train_target_temp.shape[0]) if train_target_temp[i]==opt.out_target]

    train_data_ind_temp =train_data_temp[train_ind_index,:,:]
    train_data_ood_temp =train_data_temp[train_ood_index,:,:]
    test_data_ind_temp =test_data_temp[test_ind_index,:,:]
    test_data_ood_temp =test_data_temp[test_ood_index,:,:]

    train_data_ind = nn.ZeroPad2d(2)(train_data_ind_temp).reshape(-1,1024).numpy()
    train_data_ood =nn.ZeroPad2d(2)(train_data_ood_temp).reshape(-1,1024).numpy()
    num_ood_train = train_data_ood.shape[0]
    train_data = np.concatenate((train_data_ood,train_data_ind),0)
    
    test_data_ind =nn.ZeroPad2d(2)(test_data_ind_temp).reshape(-1,1024).numpy()
    test_data_ood=[]
    test_data_ood.append(nn.ZeroPad2d(2)(test_data_ood_temp).reshape(-1,1024).numpy())
    num_ood_test = train_data_ood.shape[0]
    test_data = np.concatenate((train_data_ood,test_data_ind),0)

    train_target = train_target_temp[train_ind_index].numpy()
    test_target = test_target_temp[test_ind_index].numpy()

elif opt.char == 'feature_total':
    layer_num = 1024

    train_ind_feature = []
    test_ind_feature = []
    test_ood_feature = []
    test_ood_feature_temp = []
    num_ood=[]

    for i in [4,6,8,10,12]:
        train_ind_feature.append(np.load(os.path.join('/HDD0','ParkYH','deep_Mahalanobis_detector',opt.outf,experiment,'Features_from_layer_'+str(i)+'_'+opt.dataset+'_'+'original'+'_train_ind.npy')))
        test_ind_feature.append(np.load(os.path.join('/HDD0','ParkYH','deep_Mahalanobis_detector',opt.outf,experiment,'Features_from_layer_'+str(i)+'_'+opt.dataset+'_'+'original'+'_test_ind.npy')))
        for j in range(num_out_datasets):
            test_ood_feature_temp.append(np.load(os.path.join('/HDD0','ParkYH','deep_Mahalanobis_detector',opt.outf,experiment,'Features_from_layer_'+str(i)+'_'+out_dataset[j]+'_'+'original'+'_test_ood.npy')))
    test_ood_feature.append(np.concatenate(test_ood_feature_temp,1))
    num_ood.append(test_ood_feature[0].shape[0])

    train_data_ind = np.concatenate(train_ind_feature,1)
    test_data_ind = np.concatenate(test_ind_feature,1)
    test_data_ood = test_ood_feature
    print(train_data_ind.shape)
    print(test_data_ood[0].shape)
    
elif opt.char == 'feature_one':
    layer_num = 1024
    train_ind_feature=dict()
    test_ind_feature=dict()
    test_ood_feature=dict()
    num_ood=dict()
    for i in range(9):
        test_ood_feature[i]=[]
        num_ood[i]=[]
        train_ind_feature[i]=np.load(os.path.join('/HDD0','ParkYH','deep_Mahalanobis_detector',opt.outf,experiment,'Features_from_layer_'+str(i)+'_'+opt.dataset+'_'+layer+'_train_ind.npy'))
        test_ind_feature[i]=np.load(os.path.join('/HDD0','ParkYH','deep_Mahalanobis_detector',opt.outf,experiment,'Features_from_layer_'+str(i)+'_'+opt.dataset+'_'+layer+'_test_ind.npy'))
        print(num_out_datasets)
        for j in range(num_out_datasets):
            test_ood_feature[i].append(np.load(os.path.join('/HDD0','ParkYH','deep_Mahalanobis_detector',opt.outf,experiment,'Features_from_layer_'+str(i)+'_'+out_dataset[j]+'_'+layer+'_test_ood.npy')))
            num_ood[i].append(test_ood_feature[i][j].shape[0])
    train_data_ind = train_ind_feature
    test_data_ind = test_ind_feature
    test_data_ood = test_ood_feature
    for i in range(9):
        print(train_data_ind[i].shape)

#         train_data_ind = np.concatenate(train_ind_feature,1)
#         test_data_ind = np.concatenate(test_ind_feature,1)
#         test_data_ood = test_ood_feature

#         layer_num = train_ood_feature.shape[1]

# print('train_data_ind', train_data_ind[0].shape,train_data_ind[1].shape, train_data_ind[2].shape)
# print('test_data_ind', test_data_ind[0].shape,test_data_ind[1].shape, test_data_ind[2].shape)
# print('test_data_ood', test_data_ood[0][0].shape,test_data_ood[1][0].shape, test_data_ood[2][0].shape)


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
        if opt.h_dim6 >0:
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc5(h))
            h = self.fc6(h)
        elif opt.h_dim5 >0:
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc4(h))
            h = self.fc5(h)
        elif opt.h_dim4 >0:
            h = F.relu(self.fc3(h))
            h = self.fc4(h)
        else:
            h = self.fc3(h)
        return h
    
    
class Generator(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,h_dim5, h_dim6):
        super(Generator, self).__init__()
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
        if opt.h_dim6 >0:
            h = F.relu(self.fc6(z))
            h = F.relu(self.fc5(h))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc3(h))
        elif opt.h_dim5 >0:
            h = F.relu(self.fc5(z))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc3(h))
        elif opt.h_dim4>0:
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

# models2=dict()
# models2[0] = AE(64, 32, 16, 8,0,0,0)
# models2[1] = AE(64, 32, 16, 8,0,0,0)
# models2[2] = AE(64, 32, 16, 8,0,0,0)
# models2[3] = AE(64, 32, 16, 8,0,0,0)

# models2[4] = AE(128, 64, 32, 16,8,0,0)
# models2[5] = AE(128, 64, 32, 16,8,0,0)
# models2[6] = AE(128, 64, 32, 16,8,0,0)
# models2[7] = AE(128, 64, 32, 16,8,0,0)

# models2[8] = AE(256, 128, 64, 32, 16, 8,0)
# models2[9] = AE(256, 128, 64, 32, 16, 8,0)
# models2[10] = AE(256, 128, 64, 32, 16, 8,0)
# models2[11] = AE(256, 128, 64, 32, 16, 8,0)
# models2[12] = AE(256, 128, 64, 32, 16, 8,0)
# models2[13] = AE(256, 128, 64, 32, 16, 8,0)

# models2[14] = AE(512,256,128,64,32,8,0)
# models2[15] = AE(512,256,128,64,32,8,0)
# models2[16] = AE(512,256,128,64,32,8,0)
# models2[17] = AE(512,256,128,64,32,8,0)

# models3=dict()
# models3[0] = AE(64, 32, 16, 4,0,0,0)
# models3[1] = AE(64, 32, 16, 4,0,0,0)
# models3[2] = AE(64, 32, 16, 4,0,0,0)
# models3[3] = AE(64, 32, 16, 4,0,0,0)

# models3[4] = AE(128, 64, 32, 16,0,0,0)
# models3[5] = AE(128, 64, 32, 16,0,0,0)
# models3[6] = AE(128, 64, 32, 16,0,0,0)
# models3[7] = AE(128, 64, 32, 16,0,0,0)

# models3[8] = AE(256, 128, 64, 32, 16, 0,0)
# models3[9] = AE(256, 128, 64, 32, 16, 0,0)
# models3[10] = AE(256, 128, 64, 32, 16,0,0)
# models3[11] = AE(256, 128, 64, 32, 16, 0,0)
# models3[12] = AE(256, 128, 64, 32, 16, 0,0)
# models3[13] = AE(256, 128, 64, 32, 16, 0,0)

# models3[14] = AE(512,256,128,64,32,0,0)
# models3[15] = AE(512,256,128,64,32,0,0)
# models3[16] = AE(512,256,128,64,32,0,0)
# models3[17] = AE(512,256,128,64,32,0,0)



optimizer=dict()
schedular=dict()
for i in range(9):
    optimizer[i] = torch.optim.Adam(models[i].parameters(), opt.lr)
    schedular[i] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[i], T_max=opt.n_epochs, eta_min=0, last_epoch=-1)

# optimizer2=dict()
# schedular2=dict()
# for i in range(18):
#     optimizer2[i] = torch.optim.Adam(models2[i].parameters(), opt.lr)
#     schedular2[i] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2[i], T_max=opt.n_epochs, eta_min=0, last_epoch=-1)

# optimizer3=dict()
# schedular3=dict()
# for i in range(18):
#     optimizer3[i] = torch.optim.Adam(models3[i].parameters(), opt.lr)
#     schedular3[i] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3[i], T_max=opt.n_epochs, eta_min=0, last_epoch=-1)

   
    
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
#     models2[i].to(device)
#     models2[i].train()
#     models3[i].to(device)
#     models3[i].train()
# model.to(device)
# model.train()

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
            ckpt_name = 'layer_{}_{}_epoch_model1'.format(j,epoch)
            ckpt_path = os.path.join('BasicAE_layerwise_TEST',opt.experiment,opt.ae_name,ckpt_name + ".pth")
            torch.save(model_state, ckpt_path)

print('=== reconstruction error calculation on test data ===')
for j in range(9):
    rc_error_ind = []
    for i, data in enumerate(train_ind_loader[j]):
        data = data.cuda()
#             print(i)
        recon_error = models[j].recon_error(data)
        rc_error_ind.append(recon_error)
    rc_error_ind_total = torch.cat(rc_error_ind,0)   
    rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
    ind_score = -rc_error_ind_total_np
    l0 = open('./BasicAE_layerwise_TEST/'+opt.experiment+'/'+opt.ae_name+'/confidence_layer_{}_in_{}_epoch_{}_train.txt'.format(j,opt.dataset,epoch), 'w')
    for i in range(ind_score.shape[0]):
        l0.write("{}\n".format(ind_score[i]))
    l0.close()

    rc_error_ind = []
    for i, data in enumerate(test_ind_loader[j]):
        data = data.cuda()
#             print(i)
        recon_error = models[j].recon_error(data)
        rc_error_ind.append(recon_error)
    rc_error_ind_total = torch.cat(rc_error_ind,0)   
    rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
    ind_score = -rc_error_ind_total_np
    l1 = open('./BasicAE_layerwise_TEST/'+opt.experiment+'/'+opt.ae_name+'/confidence_layer_{}_in_{}_epoch_{}.txt'.format(j,opt.dataset,epoch), 'w')
    for i in range(ind_score.shape[0]):
        l1.write("{}\n".format(ind_score[i]))
    l1.close()

    for out_n in range(num_out_datasets):
        rc_error_ood = []
        for i, data in enumerate(test_ood_loader[j][out_n]):
            data = data.cuda()
#             print(i)
            recon_error = models[j].recon_error(data)
            rc_error_ood.append(recon_error)
        rc_error_ood_total = torch.cat(rc_error_ood,0)   
        rc_error_ood_total_np = rc_error_ood_total.detach().cpu().numpy()        

        ood_score = -rc_error_ood_total_np
        l2 = open('./BasicAE_layerwise_TEST/'+opt.experiment+'/'+opt.ae_name+'/confidence_layer_{}_out_{}_epoch_{}_model1.txt'.format(j,out_dataset[out_n],epoch), 'w')
        for i in range(ood_score.shape[0]):
            l2.write("{}\n".format(ood_score[i]))
        l2.close()


# for j in range(9):
#     for epoch in range(1, opt.n_epochs+ 1):
#         avg_loss = 0
#         step = 0
#         for i, data in enumerate(train_ind_loader[j]):
#             step += 1
#             data = data.cuda()
#     #         print(data)
#             optimizer2[j].zero_grad()
#             recon_error = models2[j].recon_error(data)
#             loss = torch.mean(recon_error)
#             loss.backward()
#             optimizer2[j].step()
#             avg_loss += loss
#             if i % 100 == 0:    
#                 print('Model for layer {} => Epoch [{}/{}] Batch [{}/{}]=> Loss: {:.5f}'.format(j,epoch, opt.n_epochs, i,len(train_ind_loader[j]), avg_loss / step))

#         if epoch % opt.ckpt_epoch == 0:
#             model_state = models2[j].state_dict()
#             #print(model_state)
#             ckpt_name = 'layer_{}_{}_epoch_model2'.format(j,epoch)
#             ckpt_path = os.path.join('BasicAE_layerwise_TEST',opt.experiment,opt.ae_name,ckpt_name + ".pth")
#             torch.save(model_state, ckpt_path)

# print('=== reconstruction error calculation on test data ===')
# for j in range(18):
#     rc_error_ind = []
#     for i, data in enumerate(train_ind_loader[j]):
#         data = data.cuda()
# #             print(i)
#         recon_error = models2[j].recon_error(data)
#         rc_error_ind.append(recon_error)
#     rc_error_ind_total = torch.cat(rc_error_ind,0)   
#     rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
#     ind_score = -rc_error_ind_total_np
#     l0 = open('./BasicAE_layerwise_TEST/'+opt.experiment+'/'+opt.ae_name+'/confidence_layer_{}_in_{}_epoch_{}_train_model2.txt'.format(j,opt.dataset,epoch), 'w')
#     for i in range(ind_score.shape[0]):
#         l0.write("{}\n".format(ind_score[i]))
#     l0.close()

#     rc_error_ind = []
#     for i, data in enumerate(test_ind_loader[j]):
#         data = data.cuda()
# #             print(i)
#         recon_error = models2[j].recon_error(data)
#         rc_error_ind.append(recon_error)
#     rc_error_ind_total = torch.cat(rc_error_ind,0)   
#     rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
#     ind_score = -rc_error_ind_total_np
#     l1 = open('./BasicAE_layerwise_TEST/'+opt.experiment+'/'+opt.ae_name+'/confidence_layer_{}_in_{}_epoch_{}_model2.txt'.format(j,opt.dataset,epoch), 'w')
#     for i in range(ind_score.shape[0]):
#         l1.write("{}\n".format(ind_score[i]))
#     l1.close()

#     for out_n in range(num_out_datasets):
#         rc_error_ood = []
#         for i, data in enumerate(test_ood_loader[j][out_n]):
#             data = data.cuda()
# #             print(i)
#             recon_error = models2[j].recon_error(data)
#             rc_error_ood.append(recon_error)
#         rc_error_ood_total = torch.cat(rc_error_ood,0)   
#         rc_error_ood_total_np = rc_error_ood_total.detach().cpu().numpy()        

#         ood_score = -rc_error_ood_total_np
#         l2 = open('./BasicAE_layerwise_TEST/'+opt.experiment+'/'+opt.ae_name+'/confidence_layer_{}_out_{}_epoch_{}_model2.txt'.format(j,out_dataset[out_n],epoch), 'w')
#         for i in range(ood_score.shape[0]):
#             l2.write("{}\n".format(ood_score[i]))
#         l2.close()




        
# for j in range(18):
#     for epoch in range(1, opt.n_epochs+ 1):
#         avg_loss = 0
#         step = 0
#         for i, data in enumerate(train_ind_loader[j]):
#             step += 1
#             data = data.cuda()
#     #         print(data)
#             optimizer3[j].zero_grad()
#             recon_error = models3[j].recon_error(data)
#             loss = torch.mean(recon_error)
#             loss.backward()
#             optimizer3[j].step()
#             avg_loss += loss
#             if i % 100 == 0:    
#                 print('Model for layer {} => Epoch [{}/{}] Batch [{}/{}]=> Loss: {:.5f}'.format(j,epoch, opt.n_epochs, i,len(train_ind_loader[j]), avg_loss / step))

#         if epoch % opt.ckpt_epoch == 0:
#             model_state = models3[j].state_dict()
#             #print(model_state)
#             ckpt_name = 'layer_{}_{}_epoch_model3'.format(j,epoch)
#             ckpt_path = os.path.join('BasicAE_layerwise_TEST',opt.experiment,opt.ae_name,ckpt_name + ".pth")
#             torch.save(model_state, ckpt_path)

# print('=== reconstruction error calculation on test data ===')
# for j in range(18):
#     rc_error_ind = []
#     for i, data in enumerate(train_ind_loader[j]):
#         data = data.cuda()
# #             print(i)
#         recon_error = models3[j].recon_error(data)
#         rc_error_ind.append(recon_error)
#     rc_error_ind_total = torch.cat(rc_error_ind,0)   
#     rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
#     ind_score = -rc_error_ind_total_np
#     l0 = open('./BasicAE_layerwise_TEST/'+opt.experiment+'/'+opt.ae_name+'/confidence_layer_{}_in_{}_epoch_{}_train_model3.txt'.format(j,opt.dataset,epoch), 'w')
#     for i in range(ind_score.shape[0]):
#         l0.write("{}\n".format(ind_score[i]))
#     l0.close()

#     rc_error_ind = []
#     for i, data in enumerate(test_ind_loader[j]):
#         data = data.cuda()
# #             print(i)
#         recon_error = models3[j].recon_error(data)
#         rc_error_ind.append(recon_error)
#     rc_error_ind_total = torch.cat(rc_error_ind,0)   
#     rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
#     ind_score = -rc_error_ind_total_np
#     l1 = open('./BasicAE_layerwise_TEST/'+opt.experiment+'/'+opt.ae_name+'/confidence_layer_{}_in_{}_epoch_{}_model3.txt'.format(j,opt.dataset,epoch), 'w')
#     for i in range(ind_score.shape[0]):
#         l1.write("{}\n".format(ind_score[i]))
#     l1.close()

#     for out_n in range(num_out_datasets):
#         rc_error_ood = []
#         for i, data in enumerate(test_ood_loader[j][out_n]):
#             data = data.cuda()
# #             print(i)
#             recon_error = models3[j].recon_error(data)
#             rc_error_ood.append(recon_error)
#         rc_error_ood_total = torch.cat(rc_error_ood,0)   
#         rc_error_ood_total_np = rc_error_ood_total.detach().cpu().numpy()        

#         ood_score = -rc_error_ood_total_np
#         l2 = open('./BasicAE_layerwise_TEST/'+opt.experiment+'/'+opt.ae_name+'/confidence_layer_{}_out_{}_epoch_{}_model3.txt'.format(j,out_dataset[out_n],epoch), 'w')
#         for i in range(ood_score.shape[0]):
#             l2.write("{}\n".format(ood_score[i]))
#         l2.close()

