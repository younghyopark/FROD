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
from data_loader_normalize import getDataLoader, getAugDataLoader
from functools import partial
from torchvision.models import resnet

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
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--out_target', default=None, help='out_target')
parser.add_argument('--aug1', default='perm', help='out_target')
parser.add_argument('--aug2', default='rot', help='out_target')
args = parser.parse_args()

print(args)

def main():
    device = torch.device("cuda")
    
    class SplitBatchNorm(torch.nn.BatchNorm2d):
        def __init__(self, num_features, num_splits, **kw):
            super().__init__(num_features, **kw)
            self.num_splits = num_splits
            
        def forward(self, input):
            N, C, H, W = input.shape
            if self.training or not self.track_running_stats:
                running_mean_split = self.running_mean.repeat(self.num_splits)
                running_var_split = self.running_var.repeat(self.num_splits)
                outcome = torch.nn.functional.batch_norm(
                    input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                    self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                    True, self.momentum, self.eps).view(N, C, H, W)
                self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
                self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
                return outcome
            else:
                return torch.nn.functional.batch_norm(
                    input, self.running_mean, self.running_var, 
                    self.weight, self.bias, False, self.momentum, self.eps)
    class ModelBase(torch.nn.Module):
        """
        Common CIFAR ResNet recipe.
        Comparing with ImageNet ResNet recipe, it:
        (i) replaces conv1 with kernel=3, str=1
        (ii) removes pool1
        """
        def __init__(self, feature_dim=128, arch='resnet18', bn_splits=8):
            super(ModelBase, self).__init__()

            # use split batchnorm
            norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else torch.nn.BatchNorm2d
            resnet_arch = getattr(resnet, arch)
            net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

            self.net = []
            for name, module in net.named_children():
                if name == 'conv1':
                    module = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                # if opt.moco_ver==2:
                #     if name == 'fc':
                #         dim_mlp = module.weight.shape[1]
                #         print(dim_mlp)
                #         module = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), module)
                if isinstance(module, torch.nn.MaxPool2d):
                    continue
                if isinstance(module, torch.nn.Linear):
                    self.net.append(torch.nn.Flatten(1))
                self.net.append(module)

            self.net = torch.nn.Sequential(*self.net)
            print(self.net)

        def forward(self, x):
            x = self.net(x)
            # note: not normalized here
            return x
    
    def create_encoder():
        emb_dim = 128
        model = ModelBase()
        model = torch.nn.DataParallel(model)
        model.to(device)
        return model

    model = create_encoder()
    print(model)


    # set the path to pre-trained model and output
    pre_trained_net = os.path.join('./trained_backbones',args.backbone_name+'.pth')
    args.outf = args.outf + args.backbone_name  + '/'
    if os.path.isdir(args.outf) == False:
        os.makedirs(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset

    # model = resnet18()
    # model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.fc = nn.Linear(512,128)
    # print(model)
    tm=torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu))
    # print(tm)
    model.load_state_dict(tm)
    model.cuda()
    print(model)
    print()
    # model.cuda()
    print('load model: ')
    print(model)
    # load dataset
    train_loader = getDataLoader(args.dataset,args.batch_size,'train')
    test_loader = getDataLoader(args.dataset,args.batch_size,'valid')

    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(2,3,32,32).cuda()
    temp_x = Variable(temp_x)
    
    # temp_list = model.feature_list(temp_x)[1]
    # num_output = len(temp_list)

    # print(num_output)
    num_output =9

    feature_list = np.empty(num_output)
    # count = 0
    # for out in temp_list:
    #     feature_list[count] = out.size(1)
    #     count += 1
        
    # print('get features for in-distribution samples')
    # for i in tqdm(range(num_output)):
    #     features = lib_extraction.moco_features(model, test_loader, i)

    #     file_name = os.path.join(args.outf, 'Features_from_layer_%s_%s_original_test_ind.npy' % (str(i), args.dataset))
    #     features = np.asarray(features, dtype=np.float32)
    #     print('layer= ',i)
    #     print(features.shape)
    #     np.save(file_name, features) 
    
    print('get features scores for transformed samples')
    aug_list = ['rot','jitter']

    for aug in aug_list:
        print(aug)
        print('')

        out_test_loader = getAugDataLoader(dataset=args.dataset,batch_size=args.batch_size,split='valid',type='loader',augmentation=aug)

        for i in tqdm(range(num_output)):
            features = lib_extraction.moco_features(model, out_test_loader, i)

            file_name = os.path.join(args.outf, 'Features_from_layer_%s_%s_original_test_aug.npy' % (str(i), aug))
            features = np.asarray(features, dtype=np.float32)
            np.save(file_name, features) 

    # print('get Mahalanobis scores for in-distribution training samples')
    # for i in tqdm(range(num_output)):
    #     features = lib_extraction.moco_features(model, train_loader, i)

    #     file_name = os.path.join(args.outf, 'Features_from_layer_%s_%s_original_train_ind.npy' % (str(i), args.dataset))
    #     features = np.asarray(features, dtype=np.float32)
    #     np.save(file_name, features) 

    
if __name__ == '__main__':
    main()
