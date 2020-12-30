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
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--ckpt_epoch',type=int,default=100)
parser.add_argument('--backbone_name','-bn', required=True, help='')
parser.add_argument("--out_target", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--dataset", default='cifar10')
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
parser.add_argument('--moco_version','-v',type=int, default=0)


opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

device = torch.device('cuda')
torch.cuda.set_device(opt.gpu)


writer = SummaryWriter(logdir=os.path.join('trained_autoencoders/VAE',opt.backbone_name))

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
layer_num=17
if opt.moco_version==1:
    layer_num=10
elif opt.moco_version==2:
    layer_num=14


train_ind_feature=dict()
test_ind_feature=dict()
test_ood_feature=dict()
num_ood=dict()
for i in range(layer_num):
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
for i in range(layer_num):
    print(train_data_ind[i].shape)


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x,):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return self.fc6(h)
    
    def recon_error(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        recon_x = self.decoder(z)
        return torch.norm((recon_x - x), dim=1)
    
    def elbo(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        recon_x = self.decoder(z)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1)
        return torch.norm((recon_x - x), dim=1) + KLD
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
    
    
def vae_loss(recon_x, x, mu, log_var):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

models=dict()
models[0] = VAE(64, 32, 16, 4)
models[1] = VAE(64, 32, 16, 4)
models[2] = VAE(64, 32, 16, 4)

models[3] = VAE(128, 64, 32, 8)
models[4] = VAE(128, 64, 32, 8)

models[5] = VAE(256, 128, 64, 16)
models[6] = VAE(256, 128, 64, 16)
models[7] = VAE(512, 256, 128, 16)
models[8] = VAE(512, 256, 128, 16)

models=dict()
models[0] = VAE(64, 32, 16, 4)
models[1] = VAE(64, 32, 16, 4)
models[2] = VAE(64, 32, 16, 4)
models[3] = VAE(64, 32, 16, 4)

models[4] = VAE(128, 64, 32, 8)
models[5] = VAE(128, 64, 32, 8)
models[6] = VAE(128, 64, 32, 8)
models[7] = VAE(128, 64, 32, 8)

models[8] = VAE(256, 128, 64, 16)
models[9] = VAE(256, 128, 64, 16)
models[10] = VAE(256, 128, 64, 16)
models[11] = VAE(256, 128, 64, 16)
models[12] = VAE(256, 128, 64, 16)
models[13] = VAE(256, 128, 64, 16)

models[14] = VAE(512, 256, 128, 16)
models[15] = VAE(512, 256, 128, 16)
models[16] = VAE(512, 256, 128, 16)

if opt.moco_version==1:
    models[9]=VAE(128, 64, 32, 8)
elif opt.moco_version==2:
    models[9] = VAE(512, 256, 128, 16)
    models[10] = VAE(512, 256, 128, 16)
    models[11] = VAE(512, 256, 128, 16)
    models[12] = VAE(512, 256, 128, 16)
    models[13] = VAE(128, 64, 32, 8)


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
    models[i].train()

print('=== train data ===')
for j in range(layer_num):
    for epoch in range(1, opt.n_epochs+ 1):
        avg_loss = 0
        step = 0
        for i, data in enumerate(train_ind_loader[j]):
            step += 1
            data = data.cuda()
    #         print(data)
            optimizer[j].zero_grad()
            elbo = models[j].elbo(data)

            loss = torch.mean(elbo)
            loss.backward()
            optimizer[j].step()
            avg_loss += loss
            if i % 100 == 0:    
                print('Model for layer {} => Epoch [{}/{}] Batch [{}/{}]=> Loss: {:.5f}'.format(j,epoch, opt.n_epochs, i,len(train_ind_loader[j]), avg_loss / step))

        if epoch % opt.ckpt_epoch == 0:
            model_state = models[j].state_dict()
            #print(model_state)
            ckpt_name = 'layer_{}_{}_epoch_model1'.format(j,epoch)
            ckpt_path = os.path.join('trained_autoencoders/VAE',opt.backbone_name,ckpt_name + ".pth")
            torch.save(model_state, ckpt_path)


# for j in range(9):
#     models[j].load_state_dict(torch.load('./trained_autoencoders/cifar10/simclr_elbo/layer_{}_500_epoch_model1.pth'.format(j)))
# epoch = 500
    
print('=== reconstruction error calculation on test data ===')
for j in range(layer_num):
    recon_error_ind = []
    elbo_ind = []
    for i, data in enumerate(train_ind_loader[j]):
        data = data.cuda()
#             print(i)
        recon_error = models[j].recon_error(data)
        recon_error_ind.append(recon_error)
        elbo = models[j].elbo(data)
        elbo_ind.append(elbo)
    rc_error_ind_total = torch.cat(recon_error_ind,0)   
    rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
    elbo_ind_total = torch.cat(elbo_ind,0)   
    elbo_ind_total_np = elbo_ind_total.detach().cpu().numpy()  
    
    ind_score = -rc_error_ind_total_np
    l0 = open('./trained_autoencoders/VAE/'+opt.backbone_name+'/confidence_layer_{}_in_{}_epoch_{}_train.txt'.format(j,opt.dataset,epoch), 'w')
    for i in range(ind_score.shape[0]):
        l0.write("{}\n".format(ind_score[i]))
    l0.close()
    
    ind_score = -elbo_ind_total_np
    l0_elbo = open('./trained_autoencoders/VAE/'+opt.backbone_name+'/elbo_layer_{}_in_{}_epoch_{}_train.txt'.format(j,opt.dataset,epoch), 'w')
    for i in range(ind_score.shape[0]):
        l0_elbo.write("{}\n".format(ind_score[i]))
    l0_elbo.close()

    recon_error_ind = []
    elbo_ind = []
    for i, data in enumerate(test_ind_loader[j]):
        data = data.cuda()
#             print(i)
        recon_error = models[j].recon_error(data)
        recon_error_ind.append(recon_error)
        elbo = models[j].elbo(data)
        elbo_ind.append(elbo)
    rc_error_ind_total = torch.cat(recon_error_ind,0)   
    rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()
    elbo_ind_total = torch.cat(elbo_ind,0)   
    elbo_ind_total_np = elbo_ind_total.detach().cpu().numpy() 
    
    ind_score = -rc_error_ind_total_np
    l1 = open('./trained_autoencoders/VAE/'+opt.backbone_name+'/confidence_layer_{}_in_{}_epoch_{}.txt'.format(j,opt.dataset,epoch), 'w')
    for i in range(ind_score.shape[0]):
        l1.write("{}\n".format(ind_score[i]))
    l1.close()
    
    ind_score = -elbo_ind_total_np
    l1_elbo = open('./trained_autoencoders/VAE/'+opt.backbone_name+'/elbo_layer_{}_in_{}_epoch_{}.txt'.format(j,opt.dataset,epoch), 'w')
    for i in range(ind_score.shape[0]):
        l1_elbo.write("{}\n".format(ind_score[i]))
    l1_elbo.close()

    for out_n in range(num_out_datasets):
        recon_error_ood = []
        elbo_ood = []
        for i, data in enumerate(test_ood_loader[j][out_n]):
            data = data.cuda()
#             print(i)
            recon_error = models[j].recon_error(data)
            recon_error_ood.append(recon_error)
            elbo = models[j].elbo(data)
            elbo_ood.append(elbo)
        rc_error_ood_total = torch.cat(recon_error_ood,0)   
        rc_error_ood_total_np = rc_error_ood_total.detach().cpu().numpy()       
        elbo_ood_total = torch.cat(elbo_ood,0)   
        elbo_ood_total_np = elbo_ood_total.detach().cpu().numpy()    

        ood_score = -rc_error_ood_total_np
        l2 = open('./trained_autoencoders/VAE/'+opt.backbone_name+'/confidence_layer_{}_out_{}_epoch_{}_model1.txt'.format(j,out_dataset[out_n],epoch), 'w')
        for i in range(ood_score.shape[0]):
            l2.write("{}\n".format(ood_score[i]))
        l2.close()
            
        ood_score = -elbo_ood_total_np
        l2_elbo = open('./trained_autoencoders/VAE/'+opt.backbone_name+'/elbo_layer_{}_out_{}_epoch_{}_model1.txt'.format(j,out_dataset[out_n],epoch), 'w')
        for i in range(ood_score.shape[0]):
            l2_elbo.write("{}\n".format(ood_score[i]))
        l2_elbo.close()