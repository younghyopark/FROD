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
#python ./autoencoder_training/vanillaAE_resnet18.py --backbone_name resnet18_simclr_permrot_svhn --gpu 1 --dataset svhn --out_dataset cifar10 -fet mean --training_layer 0
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument('--backbone_name','-bn', required=True, help='')
parser.add_argument('--gpu', type=int, required=True, help='gpu index')
parser.add_argument('--ckpt_epoch',type=int,default=100)
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
parser.add_argument('--feature_extraction_type','-fet',type=str, default='')
parser.add_argument('--layer_num','-l',type=int, default=9)
parser.add_argument('--training_layer','-tl',type=int, required=True)
parser.add_argument('--moco_version','-v',type=int, default=0)


opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

device = torch.device('cuda')
torch.cuda.set_device(opt.gpu)

writer = SummaryWriter(logdir=os.path.join('trained_autoencoders','convAE',opt.backbone_name))

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
layer_num = opt.layer_num
print('layer num', layer_num)
train_ind_feature=dict()
test_ind_feature=dict()
test_ood_feature=dict()
num_ood=dict()

for i in range(opt.training_layer,opt.training_layer+1):
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
for i in range(opt.training_layer,opt.training_layer+1):
    print(train_data_ind[i].shape)

    
class AE(nn.Module):
    """autoencoder"""
    def __init__(self, encoder, decoder):
        """
        encoder, decoder : neural networks
        """
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.own_optimizer = False

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        z = self.encoder(x)
        return z

    def recon_error(self, x):
        recon = self(x)
        recon_err = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return recon_err

    def reconstruct(self, x):
        return self(x)


    
# ConvNet desigend for 32x32 input
class ConvNet2(nn.Module):
    def __init__(self, in_chan=1, out_chan=64, nh=8, out_activation=None):
        """nh: determines the numbers of conv filters"""
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, out_chan, kernel_size=4, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max2(x)
        x = self.conv5(x)
        if self.out_activation == 'tanh':
            x = torch.tanh(x)
        elif self.out_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.out_activation == 'softmax':
            x = F.log_softmax(x, dim=1)
        return x



class DeConvNet2(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=8, out_activation=None):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet2, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_chan, nh * 16, kernel_size=4, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.conv5 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=3, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        if self.out_activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x

    
encoders = dict()
decoders = dict()

encoders[0]=ConvNet2(in_chan=64, out_chan=32, nh=4, out_activation='linear')
encoders[1]=ConvNet2(in_chan=64, out_chan=32, nh=4, out_activation='linear')
encoders[2]=ConvNet2(in_chan=64, out_chan=32, nh=4, out_activation='linear')
encoders[3]=ConvNet2(in_chan=32, out_chan=16, nh=4, out_activation='linear')
encoders[4]=ConvNet2(in_chan=32, out_chan=16, nh=4, out_activation='linear')
encoders[5]=ConvNet2(in_chan=16, out_chan=8, nh=4, out_activation='linear')
encoders[6]=ConvNet2(in_chan=16, out_chan=8, nh=4, out_activation='linear')
encoders[7]=ConvNet2(in_chan=8, out_chan=4, nh=4, out_activation='linear')
encoders[8]=ConvNet2(in_chan=8, out_chan=4, nh=4, out_activation='linear')

decoders[0]=DeConvNet2(in_chan=32, out_chan=64, nh=4, out_activation='sigmoid')
decoders[1]=DeConvNet2(in_chan=32, out_chan=64, nh=4, out_activation='sigmoid')
decoders[2]=DeConvNet2(in_chan=32, out_chan=64, nh=4, out_activation='sigmoid')
decoders[3]=DeConvNet2(in_chan=16, out_chan=32, nh=4, out_activation='sigmoid')
decoders[4]=DeConvNet2(in_chan=16, out_chan=32, nh=4, out_activation='sigmoid')
decoders[5]=DeConvNet2(in_chan=8, out_chan=16, nh=4, out_activation='sigmoid')
decoders[6]=DeConvNet2(in_chan=8, out_chan=16, nh=4, out_activation='sigmoid')
decoders[7]=DeConvNet2(in_chan=4, out_chan=8, nh=4, out_activation='sigmoid')
decoders[8]=DeConvNet2(in_chan=4, out_chan=8, nh=4, out_activation='sigmoid')


models=dict()
for i in range(9):
    models[i] = AE(encoders[i],decoders[i])


optimizer=dict()
schedular=dict()
for i in range(opt.training_layer,opt.training_layer+1):
    optimizer[i] = torch.optim.Adam(models[i].parameters(), opt.lr)
    schedular[i] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[i], T_max=opt.n_epochs, eta_min=0, last_epoch=-1)

train_ind_loader=dict()
test_ind_loader=dict()
test_ood_loader=dict()
for i in range(opt.training_layer,opt.training_layer+1):
    train_ind_loader[i] = torch.utils.data.DataLoader(torch.Tensor(train_data_ind[i]), batch_size=opt.batch_size, shuffle=True, pin_memory = True)
    test_ind_loader[i] = torch.utils.data.DataLoader(torch.Tensor(test_data_ind[i]), batch_size=opt.batch_size, shuffle=False)
    test_ood_loader[i]=[]
    for j in range(num_out_datasets):
        test_ood_loader[i].append(torch.utils.data.DataLoader(torch.Tensor(test_data_ood[i][j]), batch_size=opt.batch_size, shuffle=False))
    models[i].to(device)
    models[i].train()

if opt.resume==0:
    for j in range(opt.training_layer,opt.training_layer+1):
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
                ckpt_path = os.path.join('trained_autoencoders','convAE',opt.backbone_name,ckpt_name + ".pth")
                torch.save(model_state, ckpt_path)

if opt.resume==1:
    epoch = 1000
    for j in range(layer_num):
        ckpt_name = 'layer_{}_epoch_{}'.format(j,epoch)
        tm = torch.load(os.path.join('trained_autoencoders','convAE',opt.backbone_name,ckpt_name + ".pth"))
        print('model {} loaded'.format(j))

print('=== reconstruction error calculation on test data ===')
for j in range(opt.training_layer,opt.training_layer+1):
    print('layer {}'.format(j))
    rc_error_ind = []
    for i, data in enumerate(tqdm(train_ind_loader[j])):
        data = data.cuda()
#             print(i)
        recon_error = models[j].recon_error(data)
        rc_error_ind.append(recon_error)
    rc_error_ind_total = torch.cat(rc_error_ind,0)   
    rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
    ind_score = -rc_error_ind_total_np
    l0 = open('./trained_autoencoders/convAE/'+opt.backbone_name+'/confidence_layer_{}_in_{}_epoch_{}_{}_train.txt'.format(j,opt.dataset,epoch, opt.feature_extraction_type), 'w')
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
    l1 = open('./trained_autoencoders/convAE/'+opt.backbone_name+'/confidence_layer_{}_in_{}_epoch_{}_{}.txt'.format(j,opt.dataset,epoch, opt.feature_extraction_type), 'w')
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
        l2 = open('./trained_autoencoders/convAE/'+opt.backbone_name+'/confidence_layer_{}_out_{}_epoch_{}_{}_model1.txt'.format(j,out_dataset[out_n],epoch, opt.feature_extraction_type), 'w')
        for i in range(ood_score.shape[0]):
            l2.write("{}\n".format(ood_score[i]))
        l2.close()

