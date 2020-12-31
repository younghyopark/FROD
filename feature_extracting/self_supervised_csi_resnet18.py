"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
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
parser.add_argument('--out_dataset6', default=None, help='out_target')
parser.add_argument('--out_dataset7', default=None, help='out_target')
parser.add_argument('--out_dataset8', default=None, help='out_target')
parser.add_argument('--out_dataset9', default=None, help='out_target')
# parser.add_argument('--out_dataset6', default='place365', help='out_target')
# parser.add_argument('--out_dataset7', default='dtd', help='out_target')
# parser.add_argument('--out_dataset8', default='gaussian_noise', help='out_target')
# parser.add_argument('--out_dataset9', default='uniform_noise', help='out_target')
parser.add_argument('--feature_extraction_type','-fet', help='mean | max | min | gram_max | gram_sum', default='mean')


args = parser.parse_args()
if args.feature_extraction_type=='mean':
    from lib_extraction import get_features
elif args.feature_extraction_type=='max':
    from lib_extraction import get_features_max as get_features
elif args.feature_extraction_type=='min':
    from lib_extraction import get_features_min as get_features
elif args.feature_extraction_type=='gram_max':
    from lib_extraction import get_features_gram_max as get_features
elif args.feature_extraction_type=='gram_mean':
    from lib_extraction import get_features_gram_mean as get_features

print(args)

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
#     temp_x = torch.rand(2,3,32,32).cuda()
#     temp_x = Variable(temp_x)
    
#     temp_list = model.feature_list(temp_x)[1]
#     num_output = len(temp_list)

#     print(num_output)
#     feature_list = np.empty(num_output)
#     count = 0
#     for out in temp_list:
#         feature_list[count] = out.size(1)
#         count += 1
    num_output=12
        

    print('get {} features for in-distribution samples'.format(args.feature_extraction_type))
    for i in tqdm(range(num_output)):
        features = get_features(model, test_loader, i)
        
        file_name = os.path.join(args.outf, 'Features_from_layer_%s_%s_%s_test_ind.npy' % (str(i), args.dataset,args.feature_extraction_type))
        features = np.asarray(features, dtype=np.float32)
        print('layer= ',i)
        print(features.shape)
        np.save(file_name, features) 
    
    print('get {} features for out-of-distribution samples'.format(args.feature_extraction_type))
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
            features = get_features(model, out_test_loader, i)

            file_name = os.path.join(args.outf, 'Features_from_layer_%s_%s_%s_test_ood.npy' % (str(i), out,args.feature_extraction_type))
            features = np.asarray(features, dtype=np.float32)
            np.save(file_name, features) 

    print('get {} features for in-distribution training samples'.format(args.feature_extraction_type))
    for i in tqdm(range(num_output)):
        features = get_features(model, train_loader, i)

        file_name = os.path.join(args.outf, 'Features_from_layer_%s_%s_%s_train_ind.npy' % (str(i), args.dataset,args.feature_extraction_type))
        features = np.asarray(features, dtype=np.float32)
        np.save(file_name, features) 


if __name__ == '__main__':
    main()
