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

def get_features_max(model, test_loader, layer_index):
    model.eval()
    features = []

    for data, target in test_loader:
        
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features,_ = torch.max(out_features, 2)
        
        features.extend(out_features.detach().cpu().numpy())    
    
    return features

def get_features_min(model, test_loader, layer_index):
    model.eval()
    features = []

    for data, target in test_loader:
        
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features,_ = torch.min(out_features, 2)
        
        features.extend(out_features.detach().cpu().numpy())    
    
    return features

def get_features_gram_max(model, test_loader, layer_index):
    model.eval()
    features = []

    for data, target in test_loader:
         
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features_transpose = torch.transpose(out_features, 1,2)
        out_features = torch.matmul(out_features, torch.transpose(out_features,1,2))
        out_features,_ = torch.max(out_features, 2)
        
        features.extend(out_features.detach().cpu().numpy()) 
    
    return features

def get_features_gram_mean(model, test_loader, layer_index):
    model.eval()
    features = []

    for data, target in test_loader:
         
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features_transpose = torch.transpose(out_features, 1,2)
        out_features = torch.matmul(out_features, torch.transpose(out_features,1,2))
        out_features = torch.mean(out_features, 2)
        
        features.extend(out_features.detach().cpu().numpy()) 
    
    return features


def get_features_simclrFROD(model, test_loader, layer_index):
    model.eval()
    features = []

    for data, target in test_loader:
        
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_features(data, layer_index)
        
        features.extend(out_features.detach().cpu().numpy())    
    
    return features

def get_img(test_loader):
    features = []
    
    for data, target in test_loader:
        bs = data.size(0)
        data = data.view(bs, -1)
        features.extend(data.numpy())
    return features

def moco_features(model, test_loader, layer_index):
    model.eval()
    features = []
    net=nn.Sequential(model.module.net[0:3],*model.module.net[3],*model.module.net[4],*model.module.net[5],*model.module.net[6],model.module.net[7:])

    for data, target in tqdm(test_loader):
        
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = net[0:layer_index+1](data)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        features.extend(out_features.detach().cpu().numpy())    
    return features

    

def moco_features_ver2(model, test_loader, layer_index):
    model.eval()
    features = []
    net=nn.Sequential(model.module.net[0:3],*model.module.net[3],*model.module.net[4],*model.module.net[5],*model.module.net[6],model.module.net[7],*model.module.net[8])
    for data, target in tqdm(test_loader):
        
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = net[0:layer_index+1](data)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        features.extend(out_features.detach().cpu().numpy())    
    return features