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
from data_loader_normalize import getDataLoader

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
    def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


    def conv1x1(in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                    base_width=64, dilation=1, norm_layer=None):
            super(BasicBlock, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


    class Bottleneck(nn.Module):
        # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
        # while original implementation places the stride at the first 1x1 convolution(self.conv1)
        # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
        # This variant is also known as ResNet V1.5 and improves accuracy according to
        # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                    base_width=64, dilation=1, norm_layer=None):
            super(Bottleneck, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class ResNet(nn.Module):

        def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                    groups=1, width_per_group=64, replace_stride_with_dilation=None,
                    norm_layer=None):
            super(ResNet, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

            self.inplanes = 64
            self.dilation = 1
            if replace_stride_with_dilation is None:
                # each element in the tuple indicates if we should replace
                # the 2x2 stride with a dilated convolution instead
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

        def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

            return nn.Sequential(*layers)

        def _forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

        def forward(self, x):
            return self._forward_impl(x)

        def feature_list(self, x):
            layer=[]
            layer.append(nn.Sequential(self.conv1,self.bn1,self.relu,self.maxpool))
            for i in range(2):
                layer.append(self.layer1[i])
            for i in range(2):
                layer.append(self.layer2[i])
            for i in range(2):
                layer.append(self.layer3[i])
            for i in range(2):
                layer.append(self.layer4[i])
            layer.append(nn.Sequential(self.avgpool,self.fc))
            out_list = []
            
            out = layer[0](x)
            out_list.append(out)
            for i in range(1,9):
                out = layer[i](out)
                out_list.append(out)
            return out, out_list

        # function to extact a specific feature
        def intermediate_forward(self, x, layer_index):
            layer=[]
            layer.append(nn.Sequential(self.conv1,self.bn1,self.relu,self.maxpool))
            for i in range(2):
                layer.append(self.layer1[i])
            for i in range(2):
                layer.append(self.layer2[i])
            for i in range(2):
                layer.append(self.layer3[i])
            for i in range(2):
                layer.append(self.layer4[i])
            layer.append(nn.Sequential(self.avgpool,self.fc))
            
            out = layer[0](x)
            if layer_index==0:
                return out
            else:
                for i in range(1,9):
                    out = layer[i](out)

                    if layer_index==i:
                        return out

    def _resnet(arch, block, layers, pretrained, progress, **kwargs):
        model = ResNet(block, layers, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                progress=progress)
            model.load_state_dict(state_dict)
        return model


    def resnet18(pretrained=False, progress=True, **kwargs):
        r"""ResNet-18 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                    **kwargs)



    def resnet34(pretrained=False, progress=True, **kwargs):
        r"""ResNet-34 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)


    # set the path to pre-trained model and output
    pre_trained_net = os.path.join('./trained_backbones',args.backbone_name+'.pth')
    args.outf = args.outf + args.backbone_name  + '/'
    if os.path.isdir(args.outf) == False:
        os.makedirs(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset

    model = resnet18()
    model.x_trans_head = nn.Linear(512, 3)
    model.y_trans_head = nn.Linear(512, 3)
    model.rot_head = nn.Linear(512, 4)
    model.fc = nn.Identity()
    model.logits = nn.Linear(512,10)
    
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
