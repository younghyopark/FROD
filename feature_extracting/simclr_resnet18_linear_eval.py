"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import time


import argparse
import torch
import numpy as np
import os
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
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn')
parser.add_argument('--outf', default='./extracted_features/', help='folder to output results')
parser.add_argument('--backbone_name', required=True, help='')
parser.add_argument('--gpu', required=True, type=int, default=0, help='gpu index')
parser.add_argument('--epochs',type=int,default = 100)

args = parser.parse_args()

print(args)
def validate(encoder, classifier, val_loader, layer_index):
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            pred = classifier(encoder.intermediate_forward(images.cuda(), layer_index).flatten(1)).argmax(dim=1)
            correct += (pred.cpu() == labels).sum().item()
    return correct / len(val_loader.dataset)


class AverageMeter(object):
    r"""
    Computes and stores the average and current value.
    Adapted from
    https://github.com/pytorch/examples/blob/ec10eee2d55379f0b9c87f4b36fcf8d0723f45fc/imagenet/main.py#L359-L380
    """
    def __init__(self, name=None, fmt='.6f'):
        fmtstr = f'{{val:{fmt}}} ({{avg:{fmt}}})'
        if name is not None:
            fmtstr = name + ' ' + fmtstr
        self.fmtstr = fmtstr
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = self.sum / self.count
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        return avg

    def __str__(self):
        val = self.val
        if isinstance(val, torch.Tensor):
            val = val.item()
        return self.fmtstr.format(val=val, avg=self.avg)



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
            layer.append(nn.Sequential(self.enc.avgpool,self.enc.fc,nn.Flatten()))
            layer.append(self.projector[0:2])
            layer.append(self.projector[2])
            out_list = []
            
            out = layer[0](x)
            out_list.append(out)
            for i in range(1,12):
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
            layer.append(nn.Sequential(self.enc.avgpool,self.enc.fc,nn.Flatten()))
            layer.append(self.projector[0:2])
            layer.append(self.projector[2])
            
            out = layer[0](x)
            if layer_index==0:
                return out
            else:
                for i in range(1,12):
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
    num_layer = 12
    for layer in range(num_layer):
        print('=========layer {}=========='.format(layer))
        with torch.no_grad():
            sample, _ = train_loader.dataset[0]
            eval_numel = model.intermediate_forward(sample.unsqueeze(0).cuda(),layer).numel()        
            
        classifier = nn.Linear(eval_numel, 10).cuda()
        
        optim = torch.optim.Adam(classifier.parameters(), lr=1e-3, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.2,milestones='60,80') #github align_uniform
        loss_meter = AverageMeter('loss')
        it_time_meter = AverageMeter('iter_time')
    
        for epoch in range(args.epochs):
            loss_meter.reset()
            it_time_meter.reset()
            t0 = time.time()
            for ii, (images, labels) in enumerate(train_loader):
                optim.zero_grad()
                with torch.no_grad():
                    feats = model.intermediate_forward(images.cuda(), layer).flatten(1)
                logits = classifier(feats)
                loss = F.cross_entropy(logits, labels.to(args.gpu))
                loss_meter.update(loss, images.shape[0])
                loss.backward()
                optim.step()
                it_time_meter.update(time.time() - t0)
#                 if ii % 150 == 0:
#                     print(f"Epoch {epoch}/{args.epochs}\tIt {ii}/{len(train_loader)}\t{loss_meter}\t{it_time_meter}")
                t0 = time.time()
            scheduler.step()
            if epoch%10==0:
                val_acc = validate(model, classifier, test_loader, layer)
                print(f"Epoch {epoch}/{args.epochs}\tval_acc {val_acc*100:.4g}%")


    
if __name__ == '__main__':
    main()
