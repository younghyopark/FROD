from __future__ import print_function
import argparse
import torch
import numpy as np
import os
import lib_extraction
# from OOD_Regression_Mahalanobis import main as regression
import torchvision 
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm 
import torchvision.transforms as trn
from tqdm import tqdm
import torchvision.models as models
from data_loader_unnormalize import getDataLoader
from tensorboardX import SummaryWriter


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter


parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn')
parser.add_argument('--outf', default='./extracted_features/', help='folder to output results')
parser.add_argument('--backbone_name', required=True, help='')
parser.add_argument('--gpu', required=True, type=int, default=0, help='gpu index')
parser.add_argument('--out_target', default=None, help='out_target')
parser.add_argument('--out_dataset1', default='svhn', help='out_target')
parser.add_argument('--out_dataset2', default='imagenet_resize', help='out_target')
parser.add_argument('--out_dataset3', default='lsun_resize', help='out_target')
parser.add_argument('--out_dataset4', default='imagenet_fix', help='out_target')
parser.add_argument('--out_dataset5', default='lsun_fix', help='out_target')
parser.add_argument('--out_dataset6', default='place365', help='out_target')
parser.add_argument('--out_dataset7', default='dtd', help='out_target')
parser.add_argument('--out_dataset8', default='gaussian_noise', help='out_target')
parser.add_argument('--out_dataset9', default='uniform_noise', help='out_target')
args = parser.parse_args()

print(args)
writer = SummaryWriter(logdir=os.path.join('trained_autoencoders','vanilla_AE',args.backbone_name))


def main():
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
    
    class SimCLR_FROD(nn.Module):
        def __init__(self, base_encoder, AE, projection_dim=128):
            super().__init__()
            self.enc = base_encoder(pretrained=False)  # load model from torchvision.models without pretrained weights.
            self.feature_dim = self.enc.fc.in_features

            # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
            # See Section B.9 of SimCLR paper.
            self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.enc.maxpool = nn.Identity()
            self.enc.fc = nn.Identity()  # remove final fully connected layer.

            # Add MLP projection.
            self.projection_dim = projection_dim
            self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                           nn.ReLU(),
                                           nn.Linear(2048, projection_dim))

            self.enc_midlayers=nn.Sequential(nn.Sequential(self.enc.conv1,self.enc.bn1,self.enc.relu,self.enc.maxpool),self.enc.layer1[0],self.enc.layer1[1],self.enc.layer2[0],self.enc.layer2[1],self.enc.layer3[0],self.enc.layer3[1],self.enc.layer4[0],self.enc.layer4[1])

            self.midlayers_num=len(self.enc_midlayers)
            self.AE = nn.Sequential(AE(64, 32, 16, 8,4,0,0),
                                   AE(64, 32, 16, 8,4,0,0),
                                   AE(64, 32, 16, 8,4,0,0),
                                   AE(128, 64, 32, 16,8,4,0),
                                    AE(128, 64, 32, 16,8,4,0),
                                    AE(256, 128, 64, 32, 16, 8,4),
                                    AE(256, 128, 64, 32, 16, 8,4),
                                    AE(512,256,128,64,32,8,4),
                                    AE(512,256,128,64,32,8,4),

                                   )
        def forward(self, x):
            feature = self.enc(x)
            projection = self.projector(feature)
            return feature, projection

        def intermediate_features(self,x,index):
            out_features=self.enc_midlayers[:index+1](x)
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)

            return out_features

        def recon_error(self,x,index):
            return self.AE[index].recon_error(self.intermediate_features(x,index))

    # set the path to pre-trained model and output
    pre_trained_net = os.path.join('./trained_backbones',args.backbone_name+'.pth')
    args.outf = args.outf + args.backbone_name  + '/'
    if os.path.isdir(args.outf) == False:
        os.makedirs(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset

    model = SimCLR_FROD(torchvision.models.resnet18,AE)
    
    tm=torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu))

    model.load_state_dict(tm)
    model.cuda()
    print(model)
    print()
    # model.cuda()
    print('load model: ')
    
    # load dataset
    train_loader = getDataLoader(args.dataset,args.batch_size,'train')
    test_loader = getDataLoader(args.dataset,args.batch_size,'valid')

    # set information about feature extaction
    model.eval()
    num_output = model.midlayers_num

    print(num_output)
    feature_list = np.empty(num_output)
        
    print('get features for in-distribution samples')
    for i in tqdm(range(num_output)):
        features = lib_extraction.get_features_simclrFROD(model, test_loader, i)

        file_name = os.path.join(args.outf, 'Features_from_layer_%s_%s_original_test_ind.npy' % (str(i), args.dataset))
        features = np.asarray(features, dtype=np.float32)
        print('layer= ',i)
        print(features.shape)
        np.save(file_name, features) 
    
    print('get features scores for out-of-distribution samples')
    out_datasets_temp = [args.out_dataset1,args.out_dataset2,args.out_dataset3,args.out_dataset4,args.out_dataset5,args.out_dataset6,args.out_dataset7,args.out_dataset8,args.out_dataset9]
    out_datasets=[]
    for out in out_datasets_temp:
        if out is not None:
            out_datasets.append(out)
            
    for out in out_datasets:
        print('out')
        print('')

        out_test_loader = getDataLoader(out,args.batch_size,'valid')

        for i in tqdm(range(num_output)):
            features = lib_extraction.get_features_simclrFROD(model, out_test_loader, i)

            file_name = os.path.join(args.outf, 'Features_from_layer_%s_%s_original_test_ood.npy' % (str(i), out))
            features = np.asarray(features, dtype=np.float32)
            np.save(file_name, features) 

    print('get Mahalanobis scores for in-distribution training samples')
    for i in tqdm(range(num_output)):
        features = lib_extraction.get_features_simclrFROD(model, train_loader, i)

        file_name = os.path.join(args.outf, 'Features_from_layer_%s_%s_original_train_ind.npy' % (str(i), args.dataset))
        features = np.asarray(features, dtype=np.float32)
        np.save(file_name, features) 
    epoch = 500
    layer_num = num_output
    print('=== reconstruction error calculation on test data ===')
    for j in range(layer_num):
        print('layer {}'.format(j))
        rc_error_ind = []
        for i, (data,_) in enumerate(tqdm(train_loader)):
            data = data.cuda()
    #             print(i)
            recon_error = model.recon_error(data,j).detach()
            rc_error_ind.append(recon_error)
        rc_error_ind_total = torch.cat(rc_error_ind,0)   
        rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
        ind_score = -rc_error_ind_total_np
        l0 = open('./trained_autoencoders/vanilla_AE/'+args.backbone_name+'/confidence_layer_{}_in_{}_epoch_{}_train.txt'.format(j,args.dataset,epoch), 'w')
        for i in range(ind_score.shape[0]):
            l0.write("{}\n".format(ind_score[i]))
        l0.close()

        rc_error_ind = []
        for i, (data,_) in enumerate(tqdm(test_loader)):
            data = data.cuda()
    #             print(i)
            recon_error = model.recon_error(data,j).detach()
            rc_error_ind.append(recon_error)
        rc_error_ind_total = torch.cat(rc_error_ind,0)   
        rc_error_ind_total_np = rc_error_ind_total.detach().cpu().numpy()  
        ind_score = -rc_error_ind_total_np
        l1 = open('./trained_autoencoders/vanilla_AE/'+args.backbone_name+'/confidence_layer_{}_in_{}_epoch_{}.txt'.format(j,args.dataset,epoch), 'w')
        for i in range(ind_score.shape[0]):
            l1.write("{}\n".format(ind_score[i]))
        l1.close()

        for out_n in range(len(out_datasets)):
            out_test_loader = getDataLoader(out_datasets[out_n],args.batch_size,'valid')
            rc_error_ood = []
            for i, (data,_) in enumerate(tqdm(out_test_loader)):
                data = data.cuda()
    #             print(i)
                recon_error = model.recon_error(data,j).detach()
                rc_error_ood.append(recon_error)
            rc_error_ood_total = torch.cat(rc_error_ood,0)   
            rc_error_ood_total_np = rc_error_ood_total.detach().cpu().numpy()        

            ood_score = -rc_error_ood_total_np
            l2 = open('./trained_autoencoders/vanilla_AE/'+args.backbone_name+'/confidence_layer_{}_out_{}_epoch_{}_model1.txt'.format(j,out_datasets[out_n],epoch), 'w')
            for i in range(ood_score.shape[0]):
                l2.write("{}\n".format(ood_score[i]))
            l2.close()

if __name__ == '__main__':
    main()

