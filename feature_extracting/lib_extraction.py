from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from tqdm import tqdm

def get_features(model, test_loader, layer_index):
    model.eval()
    features = []

    for data, target in test_loader:
        
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        features.extend(out_features.detach().cpu().numpy())    
    
    return features

def get_img(test_loader):
    features = []
    
    for data, target in test_loader:
        bs = data.size(0)
        data = data.view(bs, -1)
        features.extend(data.numpy())
    return features