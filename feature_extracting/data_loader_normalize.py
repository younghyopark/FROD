import os
import numpy as np
import torch
from torchvision import datasets, transforms as trn
from torch.utils.data import DataLoader
from PIL import Image
import random

class perm_4(object):

    def __call__(self, img):
        split=4
        C, H, W = img.size()
        h = H // split #16
        w = W // split #16
        split_img = []
        for i in range(split):
            for j in range(split):
                split_img.append(img[:, i * h : (i + 1) * h, j * w : (j + 1) * w])
        split_img = torch.stack(split_img) #s*s, C, h, w

        rand = torch.rand(1, split * split)
        batch_rand_perm = (rand.argsort(dim=1))[0] # s*s

        rows = []
        for r in range(split):
            cols = []
            for c in range(split):
                cols.append(split_img[batch_rand_perm[r * split + c]])
            cols = torch.cat(cols,2)
            rows.append(cols)
        rows = torch.cat(rows,1)

        return rows
        
    def __repr__(self):
        return self.__class__.__name__+'()'

class rand_rot90(object):

    def __call__(self, img):
        img = torch.rot90(img, torch.randint(1,4,(1,))[0], [1,2])
        return img
        
    def __repr__(self):
        return self.__class__.__name__+'()'


class Place365Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, idx=0):
        """
            data_root(str) : Root directory of datasets (e.g. "/home/sr2/HDD2/Openset/")
            split_root(str) : Root directroy of split file (e.g. "/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/")
            dataset(str) : dataset name
            split(str) : ['train', 'valid', 'test']
            transform(torchvision transform) : image transform
            targets(list of str) : using targets
        """
        if idx > 32:
            raise
        self.data_root = data_root
        self.fnames = os.listdir(data_root)[idx * 10000 : (idx + 1) * 10000]
        self.transform = trn.Compose([trn.Resize((32,32)), trn.ToTensor(), trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                
            
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.data_root, fname))
        rgb= img.convert("RGB")
        img = self.transform(rgb)
        return img, 0
       
    
    def __len__(self):
        return len(self.fnames)
    
    def __str__(self):
        return "==== Place365Dataset ==== Num DATA: {}".format(len(self.fnames))
    
class DTDDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        """
            data_root(str) : Root directory of datasets (e.g. "/home/sr2/HDD2/Openset/")
            split_root(str) : Root directroy of split file (e.g. "/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/")
            dataset(str) : dataset name
            split(str) : ['train', 'valid', 'test']
            transform(torchvision transform) : image transform
            targets(list of str) : using targets
        """
        self.data_root = data_root
        self.fnames = []
        for c in os.listdir(data_root):
            class_dir = os.path.join(data_root, c)
            for f in os.listdir(class_dir):
                if '.jpg' in f:
                    self.fnames.append(os.path.join(c, f))

        random.shuffle(self.fnames)
        self.transform = trn.Compose([trn.Resize((32,32)), trn.ToTensor(), trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                
            
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.data_root, fname))
        img = self.transform(img)
        return img, 0
       
    
    def __len__(self):
        return len(self.fnames)
    
    def __str__(self):
        return "==== Place365Dataset ==== Num DATA: {}".format(len(self.fnames))
    
    
class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, tensor_name):
        self.data_root = data_root
        self.tensor_name = tensor_name
        self.data = torch.load(os.path.join(data_root, tensor_name))             
        
    def __getitem__(self, idx):
        return self.data[idx], 0
    
    def __len__(self):
        return self.data.size(0)
    
    def __str__(self):
        return "==== {} Dataset ==== Num DATA: {}".format(self.tensor_name, self.data.size(0))


def getDataLoader(dataset, batch_size, split, droot='./data',type='loader'):
    if dataset in ['cifar10']:
        mean = np.array([[0.4914, 0.4822, 0.4465]]).T
        std = np.array([[0.2023, 0.1994, 0.2010]]).T
        normalize = trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            
        transform_train = trn.Compose([
#                 trn.RandomCrop(32, padding=4),
#                 trn.RandomHorizontalFlip(),
                trn.ToTensor(),
                normalize  
            ])

        transform_test = trn.Compose([
                trn.CenterCrop(size=(32, 32)),
                trn.ToTensor(),
                normalize
            ])
   
        if split=='train':
            loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(droot, train=True, download=True,
                            transform=transform_train),
                batch_size=batch_size, shuffle=False)
        else:
            loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(droot, train=False, download=True,transform=transform_test),
                batch_size=batch_size, shuffle=False)
        print('cifar10 loaded')

    elif dataset in ['svhn']:
        mean = np.array([[0.4914, 0.4822, 0.4465]]).T
        std = np.array([[0.2023, 0.1994, 0.2010]]).T
        normalize = trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        transform_train = trn.Compose([
#                 trn.RandomCrop(32, padding=4),
#                 trn.RandomHorizontalFlip(),
                trn.ToTensor(),
                normalize
                
            ])
        transform_test = trn.Compose([
            trn.CenterCrop(size=(32, 32)),
                trn.ToTensor(),
                normalize
            ])

        if split=='train':
            loader = torch.utils.data.DataLoader(
                datasets.SVHN(droot, split="train", download=True,
                            transform=transform_train),
                batch_size=batch_size, shuffle=False)
        else:
            loader = torch.utils.data.DataLoader(
                datasets.SVHN(droot, split="test", download=True, transform=transform_test),
                batch_size=batch_size, shuffle=False)
        print('svhn loaded')

    elif dataset in ['imagenet_resize']:
        dataroot = os.path.join(droot,'Imagenet_resize')
        testsetout = datasets.ImageFolder(dataroot, transform=trn.Compose([trn.ToTensor(),trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
        print('imagenet resize loaded')

    elif dataset in ['lsun_resize']:
        dataroot = os.path.join(droot,'LSUN_resize')
        testsetout = datasets.ImageFolder(dataroot, transform=trn.Compose([trn.ToTensor(),trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
        print('lsun resize loaded')


    elif dataset in ['imagenet_fix']:
        dataroot = os.path.join(droot,'Imagenet_FIX')
        testsetout = datasets.ImageFolder(dataroot, transform=trn.Compose([trn.ToTensor(),trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
        print('imagenet fix loaded')


    elif dataset in ['lsun_fix']:
        dataroot = os.path.join(droot,'LSUN_FIX')
        testsetout = datasets.ImageFolder(dataroot, transform=trn.Compose([trn.ToTensor(),trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
        print('lsun fix loaded')

    elif dataset in ['place365']:
        dataroot = os.path.join(droot,'Place365test')
        testsetout = Place365Dataset(dataroot)
        loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
        print('places365 loaded')

    elif dataset in ['dtd']:
        dataroot = os.path.join(droot,'DTD')
        testsetout = DTDDataset(dataroot)
        loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
        print('dtd loaded')

    elif dataset in ['gaussian_noise']:
        dataroot = droot
        testsetout = TensorDataset(dataroot, 'GaussianNoise32.pth')
        loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
        print('gaussian noise loaded')

    elif dataset in ['uniform_noise']:
        dataroot = droot
        testsetout = TensorDataset(dataroot, 'UniformNoise32.pth')
        loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
        print('uniform noise loaded')

    else:
        print('nothing is loaded')

    if type=='loader':
        return loader
    else:
        return dataset




def getAugDataLoader(dataset, batch_size, split, droot='./data',type='loader',augmentation='perm4'):
    if dataset in ['cifar10']:
        mean = np.array([[0.4914, 0.4822, 0.4465]]).T
        std = np.array([[0.2023, 0.1994, 0.2010]]).T
        normalize = trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if augmentation=='perm4':
            transform_train = trn.Compose([
    #                 trn.RandomCrop(32, padding=4),
    #                 trn.RandomHorizontalFlip(),
                    trn.ToTensor(),
                    perm_4(),
                    # normalize  
                ])

            transform_test = trn.Compose([
                    trn.CenterCrop(size=(32, 32)),
                    trn.ToTensor(),
                    # normalize
                    perm_4(),
                ])
        elif augmentation =='rot':
            print('here')
            transform_train = trn.Compose([
    #                 trn.RandomCrop(32, padding=4),
    #                 trn.RandomHorizontalFlip(),
                    trn.ToTensor(),
                    rand_rot90(),
                    normalize  
                ])
            print('here2')
            transform_test = trn.Compose([
                    trn.CenterCrop(size=(32, 32)),
                    trn.ToTensor(),
                    rand_rot90(),
                    normalize
                ])
            print('here3')
        elif augmentation == 'jitter':
            transform_train = trn.Compose([
    #                 trn.RandomCrop(32, padding=4),
    #                 trn.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    trn.ToTensor(),
                    normalize  
                ])

        if split=='train':
            loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(droot, train=True, download=True,
                            transform=transform_train),
                batch_size=batch_size, shuffle=False)
        else:
            print('here4')
            loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(droot, train=False, download=True,transform=transform_test),
                batch_size=batch_size, shuffle=False)
        print('cifar10 loaded')
    else:
        assert('unsuppoted yet')
    
    if type=='loader':
        return loader
    else:
        return dataset

