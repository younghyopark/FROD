##############################################################################
### Many parts of this are a modified version of the official MoCo code ######
############### https://github.com/facebookresearch/moco #####################
##############################################################################
from torchvision import transforms, datasets
from PIL import ImageFilter
import random
import torch
from pytorch_metric_learning.utils import logging_presets
from pytorch_metric_learning import losses, miners
import record_keeper
from torchvision.models import resnet
from tqdm import tqdm
import logging
import os
from functools import partial
import argparse 
import torch.nn as nn

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--moco_ver','-v', type=int, default=1, help='gpu index')
parser.add_argument('--backbone_name','-bn', type=str)
parser.add_argument('--batch_size','-bs', type=int) 
parser.add_argument('--gpu',type=int) 




opt = parser.parse_args()

device = torch.device("cuda")
device = torch.device("cuda")

######################
### from MoCo repo ###
######################
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

######################
### from MoCo repo ###
######################
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


######################
### from MoCo repo ###
######################
def create_dataset(batch_size):
    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    # MoCo ver. 1
    if opt.moco_ver==1:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])

    # MoCo ver. 2
    elif opt.moco_ver==2:
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize
        ])

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

    train_transform = TwoCropsTransform(train_transform)

    val_transform = transforms.Compose([transforms.ToTensor(),
                                        normalize])

    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    train_dataset_for_eval = datasets.CIFAR10("./data", train=True, download=True, transform=val_transform)
    val_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=32, pin_memory=True, drop_last=True)

    train_loader_for_eval = torch.utils.data.DataLoader(
        train_dataset_for_eval, batch_size=batch_size, shuffle=False,
        num_workers=32, pin_memory=True, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=32, pin_memory=True, drop_last=False)

    return train_dataset, train_dataset_for_eval, val_dataset, train_loader, train_loader_for_eval, val_loader

######################
### from MoCo repo ###
######################
# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
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

######################
### from MoCo repo ###
######################
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
            if opt.moco_ver==2:
                if name == 'fc':
                    dim_mlp = module.weight.shape[1]
                    module = nn.Sequential(torch.nn.Flatten(1),nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, feature_dim))
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


######################
### from MoCo repo ###
######################
def copy_params(encQ, encK, m=None):
    if m is None:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    else:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)


def create_encoder():
    emb_dim = 128
    model = ModelBase()
    model = torch.nn.DataParallel(model)
    model.to(device)
    return model

#####################
### from MoCo repo ###
######################
# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, epoch, knn_k, knn_t, record_keeper):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = torch.nn.functional.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = torch.nn.functional.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            acc = total_top1 / total_num * 100
            test_bar.set_description('Test Epoch {}: Acc@1:{:.2f}%'.format(epoch, acc))

    record_keeper.update_records({"knn_monitor_accuracy": acc}, epoch, input_group_name_for_non_objects = "accuracy")
    record_keeper.save_records() 
    return acc


######################
### from MoCo repo ###
######################
# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def update_records(loss, loss_fn, optimizer, record_keeper, global_iteration):
    def optimizer_custom_attr_func(opt):
        return {"lr": opt.param_groups[0]["lr"]}
    record_these = [[{"loss": loss.item()}, {"input_group_name_for_non_objects": "loss_histories"}],
                    [{"loss_function": loss_fn}, {"recursive_types": [torch.nn.Module]}],
                    [{"optimizer": optimizer}, {"custom_attr_func": optimizer_custom_attr_func}]]
    for record, kwargs in record_these:
        record_keeper.update_records(record, global_iteration, **kwargs)


def save_model(encQ):
    model_folder = "opt.backbone_name"
    if not os.path.exists(model_folder): os.makedirs(model_folder)
    torch.save(encQ.state_dict(), "{}/encQ_best.pth".format(model_folder))


######################
### from MoCo repo ###
######################
def batch_shuffle_single_gpu(x):
    """
    Batch shuffle, for making use of BatchNorm.
    """
    # random shuffle index
    idx_shuffle = torch.randperm(x.shape[0]).cuda()

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    return x[idx_shuffle], idx_unshuffle

######################
### from MoCo repo ###
######################
def batch_unshuffle_single_gpu(x, idx_unshuffle):
    """
    Undo batch shuffle.
    """
    return x[idx_unshuffle]


def create_labels(num_pos_pairs, previous_max_label):
    # create labels that indicate what the positive pairs are
    labels = torch.arange(0, num_pos_pairs)
    labels = torch.cat((labels , labels)).to(device)
    # add an offset so that the labels do not overlap with any labels in the memory queue
    labels += previous_max_label + 1
    # we want to enqueue the output of encK, which is the 2nd half of the batch 
    enqueue_idx = torch.arange(num_pos_pairs, num_pos_pairs*2)
    return labels, enqueue_idx


def train(encQ, encK, paramK_momentum, loss_fn, optimizer, train_loader, record_keeper, global_iteration):
    encQ.train()
    pbar = tqdm(train_loader)
    for images, _ in pbar:
        previous_max_label = torch.max(loss_fn.label_memory)
        imgQ = images[0].to(device)
        imgK = images[1].to(device)

        # compute output
        encQ_out = encQ(imgQ)
        with torch.no_grad():  # no gradient to keys
            copy_params(encQ, encK, m = paramK_momentum)
            imgK, idx_unshuffle = batch_shuffle_single_gpu(imgK)
            encK_out = encK(imgK)
            encK_out = batch_unshuffle_single_gpu(encK_out, idx_unshuffle)

        all_enc = torch.cat([encQ_out, encK_out], dim=0)
        labels, enqueue_idx = create_labels(encQ_out.size(0), previous_max_label)
        loss = loss_fn(all_enc, labels, enqueue_idx = enqueue_idx)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_description("loss=%.5f" % loss.item())
        update_records(loss, loss_fn, optimizer, record_keeper, global_iteration["iter"])
        global_iteration["iter"] += 1

batch_size = opt.batch_size
lr = 0.03
paramK_momentum = 0.99
memory_size = 4096
num_epochs = 200
knn_k = 200
knn_t = 0.1

train_dataset, train_dataset_for_eval, val_dataset, \
    train_loader, train_loader_for_eval, val_loader = create_dataset(batch_size)

encQ = create_encoder()
encK = create_encoder()

# copy params from encQ into encK
copy_params(encQ, encK)

optimizer = torch.optim.SGD(encQ.parameters(), lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

###########################################################
### Set the loss function and the (optional) miner here ###
###########################################################
loss_fn = losses.CrossBatchMemory(loss = losses.NTXentLoss(temperature = 0.1),
                                  embedding_size = 128,
                                  memory_size = memory_size)

dataset_dict = {"train": train_dataset_for_eval, "val": val_dataset}
record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
hooks = logging_presets.get_hook_container(record_keeper)

# first check untrained performance
epoch = 0
best_accuracy = test(encQ, train_loader_for_eval, val_loader, epoch, knn_k, knn_t, record_keeper)

global_iteration = {"iter": 0}
for epoch in range(1, num_epochs+1):
    logging.info("Starting epoch {}".format(epoch))
    train(encQ, encK, paramK_momentum, loss_fn, optimizer, train_loader, record_keeper, global_iteration)
    curr_accuracy = test(encQ, train_loader_for_eval, val_loader, epoch, knn_k, knn_t, record_keeper)
    if curr_accuracy > best_accuracy:
        best_accuracy = curr_accuracy
        save_model(encQ)
    scheduler.step()



