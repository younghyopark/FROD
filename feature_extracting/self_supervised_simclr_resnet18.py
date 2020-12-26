"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
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

def main():
    class SimCLR(nn.Module):
        def __init__(self, base_encoder, projection_dim=128):
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

        def forward(self, x):
            feature = self.enc(x)
            projection = self.projector(feature)
            return feature, projection

        def feature_list(self, x):
            layer=[]
            layer.append(nn.Sequential(self.enc.conv1,self.enc.bn1,self.enc.relu,self.enc.maxpool))
            for i in range(2):
                layer.append(self.enc.layer1[i])
            for i in range(2):
                layer.append(self.enc.layer2[i])
            for i in range(2):
                layer.append(self.enc.layer3[i])
            for i in range(2):
                layer.append(self.enc.layer4[i])
            layer.append(nn.Sequential(self.enc.avgpool,self.enc.fc))
            out_list = []
            
            out = layer[0](x)
            out_list.append(out)
            for i in range(1,9):
                out = layer[i](out)
                out_list.append(out)
#             y = self.projector(out)
            return self.projector(self.enc(x)), out_list

        # function to extact a specific feature
        def intermediate_forward(self, x, layer_index):
            layer=[]
            layer.append(nn.Sequential(self.enc.conv1,self.enc.bn1,self.enc.relu,self.enc.maxpool))
            for i in range(2):
                layer.append(self.enc.layer1[i])
            for i in range(2):
                layer.append(self.enc.layer2[i])
            for i in range(2):
                layer.append(self.enc.layer3[i])
            for i in range(2):
                layer.append(self.enc.layer4[i])
            layer.append(nn.Sequential(self.enc.avgpool,self.enc.fc))
            
            out = layer[0](x)
            if layer_index==0:
                return out
            else:
                for i in range(1,9):
                    out = layer[i](out)

                    if layer_index==i:
                        return out
#             y = self.projector(out)


    # set the path to pre-trained model and output
    pre_trained_net = os.path.join('./trained_backbones',args.backbone_name+'.pth')
    args.outf = args.outf + args.backbone_name  + '/'
    if os.path.isdir(args.outf) == False:
        os.makedirs(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset

    model = SimCLR(models.resnet18)
    
    tm=torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu))

    model.load_state_dict(tm)
    model.cuda()
    print(model.enc)
    print()
    # model.cuda()
    print('load model: ')
    
    # load dataset
    train_loader = getDataLoader(args.dataset,args.batch_size,'train')
    test_loader = getDataLoader(args.dataset,args.batch_size,'valid')

    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(2,3,32,32).cuda()
    temp_x = Variable(temp_x)
    
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)

    print(num_output)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
        
    print('get features for in-distribution samples')
    for i in tqdm(range(num_output)):
        features = lib_extraction.get_features(model, test_loader, i)

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
            features = lib_extraction.get_features(model, out_test_loader, i)

            file_name = os.path.join(args.outf, 'Features_from_layer_%s_%s_original_test_ood.npy' % (str(i), out))
            features = np.asarray(features, dtype=np.float32)
            np.save(file_name, features) 

    print('get Mahalanobis scores for in-distribution training samples')
    for i in tqdm(range(num_output)):
        features = lib_extraction.get_features(model, train_loader, i)

        file_name = os.path.join(args.outf, 'Features_from_layer_%s_%s_original_train_ind.npy' % (str(i), args.dataset))
        features = np.asarray(features, dtype=np.float32)
        np.save(file_name, features) 

    
if __name__ == '__main__':
    main()
