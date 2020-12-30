import hydra
from omegaconf import DictConfig
import logging

import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34
from torchvision import transforms
import torch.nn as nn

from models import SimCLR, SimCLR_FROD
from tqdm import tqdm


logger = logging.getLogger(__name__)
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CIFAR10Pair(CIFAR10):
    """Generate mini-batche pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair

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
    
def nt_xent(x, t=0.5):
    x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


@hydra.main(config_path='simclr_config.yml')
def train(args: DictConfig) -> None:
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          get_color_distortion(s=0.5),
                                          transforms.ToTensor()])
#     original_train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
#                                           transforms.RandomHorizontalFlip(p=0.5),
#                                           transforms.ToTensor()])
    ood_train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.ToTensor(),
                                          transforms.RandomApply([perm_4()],p=0.5),
                                          rand_rot90()])

    data_dir = hydra.utils.to_absolute_path(args.data_dir)  # get absolute path of data dir
    
    train_set = CIFAR10Pair(root=data_dir,
                            train=True,
                            transform=train_transform,
                            download=True)
#     original_train_set = CIFAR10(root=data_dir,
#                             train=True,
#                             transform=original_train_transform,
#                             download=True)
    
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              drop_last=True)
#     original_train_loader = DataLoader(original_train_set,
#                               batch_size=args.batch_size,
#                               shuffle=True,
#                               num_workers=args.workers,
#                               drop_last=True)
    
    ood_train_set = CIFAR10(root=data_dir,
                            train=True,
                            transform=ood_train_transform,
                            download=True)
    ood_train_loader = DataLoader(ood_train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              drop_last=True)

    # Prepare model
    assert args.backbone in ['resnet18', 'resnet34']
    base_encoder = eval(args.backbone)
    model = SimCLR_FROD(base_encoder, AE, projection_dim=args.projection_dim).cuda()
    logger.info('Base model: {}'.format(args.backbone))
    logger.info('feature dim: {}, projection dim: {}'.format(model.feature_dim, args.projection_dim))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            args.learning_rate,  # lr_lambda computes multiplicative factor
            1e-3))

    # SimCLR training
    model.train()
    for epoch in range(1, args.epochs + 1):
        
        if epoch>args.warmup_epoch and epoch%args.FROD_epoch==1: print('Training Autoencoder for this epoch!')
        
        loss_meter = AverageMeter("total loss")
        simclr_loss_meter = AverageMeter("SimCLR loss")
        ae_ind_loss_meter = AverageMeter("AE_IND loss")
        ae_ood_loss_meter = AverageMeter("AE_OOD loss")
        train_bar = tqdm(train_loader)
        for count,((x, y),(x_ood,y_ood)) in enumerate(zip(train_bar,ood_train_loader)):
            sizes = x.size()
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda(non_blocking=True)
            x_ood = x_ood.cuda()
            
            optimizer.zero_grad()
            feature, rep = model(x)
            loss = nt_xent(rep, args.temperature)
            if epoch==1 and count==0:
                loss1=(torch.mean(model.recon_error(x,0)))/model.midlayers_num
                loss2=(torch.mean(model.recon_error(x_ood,0)))/model.midlayers_num
            if epoch>args.warmup_epoch and epoch%args.FROD_epoch==1:
                for i in range(model.midlayers_num):
    #             i= model.midlayers_num-1
                    loss1=(torch.mean(model.recon_error(x,i)))/model.midlayers_num
                    loss2=(torch.mean(model.recon_error(x_ood,i)))/model.midlayers_num
                    loss+=args.AE_coef*(3*loss1-loss2)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), x.size(0))
            simclr_loss_meter.update(nt_xent(rep, args.temperature).item(), x.size(0))
            ae_ind_loss_meter.update(loss1.item(), x.size(0))
            ae_ood_loss_meter.update(loss2.item(), x.size(0))

            train_bar.set_description("Train epoch {}, Total loss: {:.4f}, SimCLR loss: {:.4f}, AE IND loss: {:.4f}, AE OOD loss: {:.4f}".format(epoch, loss_meter.avg,simclr_loss_meter.avg, ae_ind_loss_meter.avg, ae_ood_loss_meter.avg))

        # save checkpoint very log_interval epochs
        if epoch % args.log_interval == 0:
#             logger.info("Train epoch {}, Total loss: {:.4f}, SimCLR loss: {:.4f}, AE loss: {:.4f}".format(epoch, loss_meter.avg,simclr_loss_meter.avg, ae_loss_meter.avg))
            torch.save(model.state_dict(), '{}.pth'.format(args.bn))


if __name__ == '__main__':
    train()



