import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torch.nn as nn
from tqdm import tqdm
import argparse
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import os


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument('--gpu', type=int, required=True, help='gpu index')
parser.add_argument('--batch_size', type=int, default=128, help='bs')
parser.add_argument('--backbone_name','-bn', type=str, required=True, help='bs')
parser.add_argument('--margin', type=float, default=1.0, help='bs')

opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

device = torch.device('cuda')
torch.cuda.set_device(opt.gpu)

def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

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


pos_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      get_color_distortion(s=0.5),
                                      transforms.ToTensor()])

pos_dataset = CIFAR10('./data', train=True, download=True,
                             transform=pos_transform)

neg_perm_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      get_color_distortion(s=0.5),
                                      transforms.ToTensor(),
                                      perm_4()])

neg_perm_dataset = CIFAR10('./data', train=True, download=True,
                             transform=neg_perm_transform)

neg_rot_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      get_color_distortion(s=0.5),
                                      transforms.ToTensor(),
                                      rand_rot90()])

neg_rot_dataset = CIFAR10('./data', train=True, download=True,
                             transform=neg_rot_transform)

class SiameseCIFAR10(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, pos_dataset, neg_dataset):
        self.pos_dataset = pos_dataset
        self.neg_dataset = neg_dataset
        
    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        if target == 1:
            pos_index = np.random.randint(0,50000)
            img1 = self.pos_dataset.__getitem__(index)[0]
            img2 = self.pos_dataset.__getitem__(pos_index)[0]
        else:
            neg_index = np.random.randint(0,50000)
            img1 = self.pos_dataset.__getitem__(index)[0]
            img2 = self.neg_dataset.__getitem__(neg_index)[0]
        
        return (img1, img2), target

    def __len__(self):
        return len(self.pos_dataset)
    
siamese_train_dataset = SiameseCIFAR10(pos_dataset,neg_rot_dataset) # Returns pairs of images and target same/different
batch_size = opt.batch_size
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

# Set up the network and training parameters
from torchvision.models import resnet18

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


margin = opt.margin
embedding_net = resnet18()
embedding_net.fc= nn.Identity()
model = SiameseNet(embedding_net)
model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = opt.n_epochs
log_interval = 100

def fit(train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        if epoch%5==0:
            model_state = model.embedding_net.state_dict()
            #print(model_state)
            ckpt_name = '{}_epoch_{}'.format(opt.backbone_name,epoch)
            ckpt_path = os.path.join('trained_backbones','testing_phase',ckpt_name + ".pth")
            torch.save(model_state, ckpt_path)
        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

fit(siamese_train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
