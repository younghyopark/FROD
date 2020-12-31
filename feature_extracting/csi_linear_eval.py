"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""


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
from abc import *




parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn')
parser.add_argument('--outf', default='./extracted_features/', help='folder to output results')
parser.add_argument('--backbone_name', required=True, help='')
parser.add_argument('--gpu', required=True, type=int, default=0, help='gpu index')
# parser.add_argument('--out_dataset6', default='place365', help='out_target')
# parser.add_argument('--out_dataset7', default='dtd', help='out_target')
# parser.add_argument('--out_dataset8', default='gaussian_noise', help='out_target')
# parser.add_argument('--out_dataset9', default='uniform_noise', help='out_target')
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
    
    class NormalizeLayer(nn.Module):
        """
        In order to certify radii in original coordinates rather than standardized coordinates, we
        add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
        layer of the classifier rather than as a part of preprocessing as is typical.
        """

        def __init__(self):
            super(NormalizeLayer, self).__init__()

        def forward(self, inputs):
            return (inputs - 0.5) / 0.5

    class BaseModel(nn.Module, metaclass=ABCMeta):
        def __init__(self, last_dim, num_classes=10, simclr_dim=128):
            super(BaseModel, self).__init__()
            self.linear = nn.Linear(last_dim, num_classes)
            self.simclr_layer = nn.Sequential(
                nn.Linear(last_dim, last_dim),
                nn.ReLU(),
                nn.Linear(last_dim, simclr_dim),
            )
            self.shift_cls_layer = nn.Linear(last_dim, 4)
            self.joint_distribution_layer = nn.Linear(last_dim, 4 * num_classes)

        @abstractmethod
        def penultimate(self, inputs, all_features=False):
            pass

        def forward(self, inputs, penultimate=False, simclr=False, shift=False, joint=False):
            _aux = {}
            _return_aux = False

            features = self.penultimate(inputs)

            output = self.linear(features)

            if penultimate:
                _return_aux = True
                _aux['penultimate'] = features

            if simclr:
                _return_aux = True
                _aux['simclr'] = self.simclr_layer(features)

            if shift:
                _return_aux = True
                _aux['shift'] = self.shift_cls_layer(features)

            if joint:
                _return_aux = True
                _aux['joint'] = self.joint_distribution_layer(features)

            if _return_aux:
                return output, _aux

            return output
    
        def intermediate_forward(self, inputs, layer_index, penultimate=False, simclr=False, shift=False, joint=False):
            _aux = {}
            _return_aux = False

            features = self.penultimate(inputs)

            output = self.linear(features)

            if penultimate:
                _return_aux = True
                _aux['penultimate'] = features

            if simclr:
                _return_aux = True
                _aux['simclr'] = self.simclr_layer(features)

            if shift:
                _return_aux = True
                _aux['shift'] = self.shift_cls_layer(features)

            if joint:
                _return_aux = True
                _aux['joint'] = self.joint_distribution_layer(features)

            if _return_aux:
                return output, _aux

            return output
        
    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = conv3x3(in_planes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


    class PreActBlock(nn.Module):
        '''Pre-activation version of the BasicBlock.'''
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(PreActBlock, self).__init__()
            self.conv1 = conv3x3(in_planes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                )

        def forward(self, x):
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out += shortcut
            return out


    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


    class PreActBottleneck(nn.Module):
        '''Pre-activation version of the original Bottleneck module.'''
        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
            super(PreActBottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                )

        def forward(self, x):
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out = self.conv3(F.relu(self.bn3(out)))
            out += shortcut
            return out


    class ResNet(BaseModel):
        def __init__(self, block, num_blocks, num_classes=10):
            last_dim = 512 * block.expansion
            super(ResNet, self).__init__(last_dim, num_classes)

            self.in_planes = 64
            self.last_dim = last_dim

            self.normalize = NormalizeLayer()

            self.conv1 = conv3x3(3, 64)
            self.bn1 = nn.BatchNorm2d(64)

            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            simclr_dim=128
            self.simclr_layer = nn.Sequential(
                nn.Linear(last_dim, last_dim),
                nn.ReLU(),
                nn.Linear(last_dim, simclr_dim),
            )
            self.fc = nn.Identity()

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def penultimate(self, x, all_features=False):
            out_list = []

            out = self.normalize(x)
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
            out_list.append(out)

            out = self.layer1[0](out)
            out_list.append(out)
            out = self.layer1[1](out)
            out_list.append(out)
            out = self.layer2[0](out)
            out_list.append(out)
            out = self.layer2[1](out)
            out_list.append(out)

            out = self.layer3[0](out)
            out_list.append(out)
            out = self.layer3[1](out)
            out_list.append(out)

            out = self.layer4[0](out)
            out_list.append(out)
            out = self.layer4[1](out)
            out_list.append(out)


            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)

            if all_features:
                return out, out_list
            else:
                return out
        
        def penultimate_vector(self, x, all_features=False):
            out_list = []
            
            out = self.normalize(x)
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
            bs, c, _, _ = out.size()
            out_list.append(torch.mean(out.view(bs, c, -1), 2))
            out = self.layer1[0](out)
            bs, c, _, _ = out.size()
            out_list.append(torch.mean(out.view(bs, c, -1), 2))
            out = self.layer1[1](out)
            bs, c, _, _ = out.size()
            out_list.append(torch.mean(out.view(bs, c, -1), 2))

            out = self.layer2[0](out)
            bs, c, _, _ = out.size()
            out_list.append(torch.mean(out.view(bs, c, -1), 2))
            out = self.layer2[1](out)
            bs, c, _, _ = out.size()
            out_list.append(torch.mean(out.view(bs, c, -1), 2))

            out = self.layer3[0](out)
            bs, c, _, _ = out.size()
            out_list.append(torch.mean(out.view(bs, c, -1), 2))
            out = self.layer3[1](out)
            bs, c, _, _ = out.size()
            out_list.append(torch.mean(out.view(bs, c, -1), 2))

            out = self.layer4[0](out)
            bs, c, _, _ = out.size()
            out_list.append(torch.mean(out.view(bs, c, -1), 2))
            out = self.layer4[1](out)
            bs, c, _, _ = out.size()
            out_list.append(torch.mean(out.view(bs, c, -1), 2))

            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)

            if all_features:
                return out, out_list
            else:
                return out
                 
        def intermediate_forward(self, data, layer_index):
            out = self.normalize(data)
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
            if layer_index == 0:
                return out
            out = self.layer1[0](out)
            if layer_index == 1:
                return out
            out = self.layer1[1](out)
            if layer_index == 2:
                return out
            out = self.layer2[0](out)
            if layer_index == 3:
                return out
            out = self.layer2[1](out)
            if layer_index == 4:
                return out
            out = self.layer3[0](out)
            if layer_index == 5:
                return out
            out = self.layer3[1](out)
            if layer_index == 6:
                return out
            out = self.layer4[0](out)
            if layer_index == 7:
                return out
            out = self.layer4[1](out)
            if layer_index == 8:
                return out
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            if layer_index == 9:
                return out
            out = self.simclr_layer[0:2](out)
            if layer_index == 10:
                return out
            out = self.simclr_layer[2](out)
            if layer_index ==11 :
                return out
            
    def ResNet18():
        return ResNet(BasicBlock, [2,2,2,2], num_classes=10)

    def ResNet34():
        return ResNet(BasicBlock, [3,4,6,3], num_classes=10)

    def ResNet50():
        return ResNet(Bottleneck, [3,4,6,3], num_classes=10)


    
    # set the path to pre-trained model and output
    pre_trained_net = os.path.join('./trained_backbones',args.backbone_name+'.pth')
    args.outf = args.outf + args.backbone_name  + '/'
    if os.path.isdir(args.outf) == False:
        os.makedirs(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset

    model = ResNet18()
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    print(pre_trained_net)
    pretrained_dict = torch.load(pre_trained_net)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()

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
