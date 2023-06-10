"""
Train and Test CNN

Store model in disk

Pre-trained models are then used for GAN evaluation (i.e., InceptionV3)

"""


import argparse
import shutil
import os
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import autograd
from torchvision.utils import save_image
import csv
from models import *
from tqdm import tqdm
import gc
import h5py

from utils import *

#############################
# Settings
#############################

parser = argparse.ArgumentParser(description='Pre-train CNNs')
parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--inception_path', type=str, default='')
parser.add_argument('--CNN', type=str, default='InceptionV3',
                    help='CNN for feature extractor or evaluation')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train CNNs (default: 200)')
parser.add_argument('--resume_epoch', type=int, default=0, metavar='N',
                    help='resume training.')
parser.add_argument('--save_freq', type=int, default=25, metavar='N',
                    help='freq to save ckpt.')
parser.add_argument('--batch_size_train', type=int, default=256, metavar='N',
                    help='input batch size for training')
parser.add_argument('--batch_size_test', type=int, default=100, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--lr_base', type=float, default=0.1,
                    help='learning rate, default=0.1')
parser.add_argument('--lr_decay_factor', type=float, default=0.1)
parser.add_argument('--lr_decay_epochs', type=str, default='100_150', help='decay lr at which epoch; separate by _')
parser.add_argument('--weight_dacay', type=float, default=5e-4,
                    help='Weigth decay, default=1e-4')
parser.add_argument('--seed', type=int, default=2020, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--transform', action='store_true', default=False,
                    help='flip or crop images for CNN training')
parser.add_argument('--num_classes', type=int, default=100, metavar='N',
                    help='number of classes')
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()


# random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

## lr decay scheme
lr_decay_epochs = (args.lr_decay_epochs).split("_")
lr_decay_epochs = [int(epoch) for epoch in lr_decay_epochs]

# directories for checkpoints
save_models_folder = os.path.join(args.root_path, 'output/saved_models')
os.makedirs(save_models_folder,exist_ok=True)

save_ckpts_in_train = os.path.join(save_models_folder, 'ckpts_in_train_eval_{}'.format(args.CNN))
os.makedirs(save_ckpts_in_train, exist_ok=True)

path_torch_home = os.path.join(args.root_path, 'torch_cache')
os.makedirs(path_torch_home, exist_ok=True)
os.environ['TORCH_HOME'] = path_torch_home


###########################################################################################################
# Data loader
###########################################################################################################
# data loader
trainset_h5py_file = args.data_path + '/ImageNet_128x128_100Class.h5'
hf = h5py.File(trainset_h5py_file, 'r')
images_train = hf['images_train'][:]
labels_train = hf['labels_train'][:]
images_test = hf['images_valid'][:]
labels_test = hf['labels_valid'][:]
hf.close()


### compute the mean and std for normalization
# assert images_train.shape[1]==3
# train_means = []
# train_stds = []
# for i in range(3):
#     images_i = images_train[:,i,:,:]
#     images_i = images_i/255.0
#     train_means.append(np.mean(images_i))
#     train_stds.append(np.std(images_i))
# ## for i
train_means = (0.5, 0.5, 0.5)
train_stds = (0.5, 0.5, 0.5) #in InceptionV3.py, images (in [-1,1]) will be re-normalized by using ImageNet constants

if args.transform:
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(train_means, train_stds),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_means, train_stds),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_means, train_stds),
])


trainset = IMGs_dataset(images_train, labels_train, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)

testset = IMGs_dataset(images_test, labels_test, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)


###########################################################################################################
# Training functions
###########################################################################################################

#initialize CNNs
def net_initialization(Pretrained_CNN_Name, num_classes=args.num_classes, ckpt_path=args.inception_path):

    # net = Inception3(num_classes=num_classes, aux_logits=True, transform_input=False)

    net = inception_v3(pretrained = True, ckpt_path=ckpt_path, num_classes = num_classes, transform_input=True) #use transform_input=True

    net_name = 'PreCNNForEval_' + Pretrained_CNN_Name #get the net's name

    net = nn.DataParallel(net).cuda()

    return net, net_name

#adjust CNN learning rate
def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate """
    lr = args.lr_base

    num_decays = len(lr_decay_epochs)
    for decay_i in range(num_decays):
        if epoch >= lr_decay_epochs[decay_i]:
            lr = lr * args.lr_decay_factor
        #end if epoch
    #end for decay_i
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_CNN():

    if args.resume_epoch>0:
        save_file = save_ckpts_in_train + "/{}_checkpoint_epoch_{}.pth".format(args.CNN, args.resume_epoch)
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])

    start_time = timeit.default_timer()
    for epoch in range(args.resume_epoch, args.epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate(optimizer, epoch)
        for batch_idx, (batch_train_images, batch_train_labels) in enumerate(trainloader):

            if args.CNN == "InceptionV3":
                batch_train_images = nn.functional.interpolate(batch_train_images, size = (299,299), scale_factor=None, mode='bilinear', align_corners=False)

            batch_train_images = batch_train_images.type(torch.float).cuda()
            batch_train_labels = batch_train_labels.type(torch.long).cuda()

            if len(batch_train_images)==trainloader.batch_size: #skip the last iteration in each epoch

                #Forward pass
                outputs,_ = net(batch_train_images)
                loss = criterion(outputs, batch_train_labels)

                #backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().item()
        #end for batch_idx
        test_acc = test_CNN(False)

        ## save ckpt during training
        if ((epoch+1)%args.save_freq==0 or (epoch+1) == args.epochs) :
            save_file = save_ckpts_in_train + "/{}_checkpoint_epoch_{}.pth".format(args.CNN, epoch+1)
            torch.save({
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)

        print('CNN: [epoch %d/%d] train_loss:%.3f, test_acc:%.3f, Time:%.3f' % (epoch+1, args.epochs, train_loss/(batch_idx+1), test_acc, timeit.default_timer()-start_time))
    #end for epoch

    return net, optimizer


def test_CNN(verbose=True):

    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            if args.CNN == "InceptionV3":
                images = nn.functional.interpolate(images, size = (299,299), scale_factor=None, mode='bilinear', align_corners=False)
            images = images.type(torch.float).cuda()
            labels = labels.type(torch.long).cuda()
            outputs,_ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        if verbose:
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100.0 * correct / total))
    return 100.0 * correct / total


###########################################################################################################
# Training and Testing
###########################################################################################################
# model initialization
net, net_name = net_initialization(args.CNN, num_classes=args.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = args.lr_base, momentum= 0.9, weight_decay=args.weight_dacay)

filename_ckpt = save_models_folder + '/ckpt_' + net_name + '_epoch_' + str(args.epochs) +  '_SEED_' + str(args.seed) + '_Transformation_' + str(args.transform) + '_finetuned.pth'

# training
if not os.path.isfile(filename_ckpt):
    # TRAIN CNN
    print("\n Begin training CNN: ")
    start = timeit.default_timer()
    net, optimizer = train_CNN()
    stop = timeit.default_timer()
    print("Time elapses: {}s".format(stop - start))
    # save model
    torch.save({
    'net_state_dict': net.state_dict(),
    }, filename_ckpt)
else:
    print("\n Ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(filename_ckpt)
    net.load_state_dict(checkpoint['net_state_dict'])

#testing
checkpoint = torch.load(filename_ckpt)
net.load_state_dict(checkpoint['net_state_dict'])
_ = test_CNN(True)
torch.cuda.empty_cache()
