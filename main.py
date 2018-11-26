# -*- coding: utf-8 -*-
# @Time    : 2018/8/28 10:12 PM
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn


# Base
import sys
import os
import time
import random
import copy
import numpy as np
from multiprocessing import Process

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import visdom
from colorama import Fore

# Mine
import config as cf
from vgg import vgg16


def toHalf(x):
    return x.half()


class GaGrad:
    def __init__(self):
        # hyper-parameter
        self.batch_size = 256
        self.lr = 1e-2
        self.num_epochs = 160

        # data
        self.train_set = None
        self.test_set = None
        self.train_loader = None
        self.test_loader = None

        # net
        # self.net = vgg16(num_classes=10).cuda()
        self.net = vgg16(num_classes=10).half().cuda()
        self.init_state_dict = copy.deepcopy(self.net.state_dict())
        self.criterion = nn.CrossEntropyLoss()

        # show
        self.show_log = False
        self.show_vis = False

        # visdom
        if self.show_vis:
            self.vis = visdom.Visdom(port=8096, env='vgg16-train-test-diff-random-seed')

        # save date
        self.train_batch_loss_list = []
        self.train_batch_acc_list = []
        self.train_epoch_loss_list = []
        self.train_epoch_acc_list = []

        self.val_epoch_loss_list = []
        self.val_epoch_acc_list = []

        self.params_l2_norm_list = []

    def load_data(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
        ])  # meanstd transformation

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
        ])

        self.train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        self.test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_set, batch_size=3000, shuffle=False, num_workers=4)

    def train(self, epoch):
        self.net.train()
        total_train_loss = 0
        total_num = 0
        total_train_correct = 0

        cur_lr = cf.learning_rate(self.lr, epoch)
        self.optimizer = optim.SGD(self.net.parameters(), lr=cur_lr, momentum=0.9, weight_decay=5e-4)

        if self.show_log:
            print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, cur_lr))

        self.train_batch_loss_list = []
        self.train_batch_acc_list = []
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = inputs.half().cuda(), targets.cuda()

            self.optimizer.zero_grad()
            outputs = self.net(inputs)  # Forward Propagation
            loss = self.criterion(outputs, targets)  # Loss
            loss.backward()  # Backward Propagation
            self.optimizer.step()  # Optimizer update

            # loss
            train_loss = loss.item()
            total_train_loss += train_loss
            self.train_batch_loss_list.append(train_loss)

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_num += targets.size(0)
            train_correct = predicted.eq(targets.data).cpu().sum().item()
            total_train_correct += train_correct
            train_acc = train_correct / targets.size(0)
            self.train_batch_acc_list.append(train_acc)

            # update visdom
            if self.show_vis:
                self.vis.line(np.array(self.train_batch_loss_list), X=np.arange(len(self.train_batch_loss_list)),
                              win='train_batch_loss', opts={'title': 'train_batch_loss'})
                self.vis.line(np.array(self.train_batch_acc_list), X=np.arange(len(self.train_batch_acc_list)),
                              win='train_batch_acc', opts={'title': 'train_batch_acc'})

            # update output
            if self.show_log:
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                                 % (epoch, self.num_epochs,
                                    batch_idx + 1, (len(self.train_set) // self.batch_size) + 1,
                                    train_loss, 100 * train_acc))
                sys.stdout.flush()

        # total loss and accuracy
        epoch_average_train_loss = total_train_loss / len(self.train_loader)
        epoch_average_train_acc = total_train_correct / total_num

        self.train_epoch_loss_list.append(epoch_average_train_loss)
        self.train_epoch_acc_list.append(epoch_average_train_acc)

        # l2 norm of all parameters
        square_params_sum = 0
        for param in self.net.parameters():
            square_params_sum += (param ** 2).sum()
        params_l2_norm = (square_params_sum ** 0.5).item()
        self.params_l2_norm_list.append(params_l2_norm)

        # update visdom
        if self.show_vis:
            self.vis.line(np.array(self.train_epoch_loss_list), X=np.arange(len(self.train_epoch_loss_list)),
                          win='train_epoch_loss', opts={'title': 'train_epoch_loss'})
            self.vis.line(np.array(self.train_epoch_acc_list), X=np.arange(len(self.train_epoch_acc_list)),
                          win='train_epoch_acc', opts={'title': 'train_epoch_acc'})
            self.vis.line(np.array(self.params_l2_norm_list), X=np.arange(len(self.params_l2_norm_list)),
                          win='params_l2_norm', opts={'title': 'params_l2_norm'})

    def val(self, epoch):
        self.net.eval()
        total_val_loss = 0
        total_num = 0
        total_val_correct = 0

        if self.show_log:
            print('\n=> Testing Epoch #%d' % epoch)
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = self.net(inputs)  # Forward Propagation
            loss = self.criterion(outputs, targets)  # Loss

            # loss
            train_loss = loss.item()
            total_val_loss += train_loss

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_num += targets.size(0)
            val_correct = predicted.eq(targets.data).cpu().sum().item()
            total_val_correct += val_correct

        epoch_average_val_loss = total_val_loss / len(self.test_loader)
        epoch_average_val_acc = total_val_correct / total_num

        self.val_epoch_loss_list.append(epoch_average_val_loss)
        self.val_epoch_acc_list.append(epoch_average_val_acc)

        # update visdom
        if self.show_vis:
            self.vis.line(np.array(self.val_epoch_loss_list), X=np.arange(len(self.val_epoch_loss_list)),
                          win='val_epoch_loss', opts={'title': 'val_epoch_loss'})
            self.vis.line(np.array(self.val_epoch_acc_list), X=np.arange(len(self.val_epoch_acc_list)),
                          win='val_epoch_acc', opts={'title': 'val_epoch_acc'})

    def run(self):
        total_time = 0
        k80_score = 0.005325219

        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.train(epoch)
            epoch_time = time.time() - start_time
            total_time += epoch_time
            print('current time: %s, average time: %s' % (epoch_time, total_time / (epoch + 1)))
            print(Fore.GREEN, 'current score: %s, average score: %s' % (1 / epoch_time / k80_score, 1 / (total_time / (epoch + 1)) / k80_score), Fore.RESET)
            # self.val(epoch)

        print('total time: %s' % total_time)

    def save_checkpoint(self, filename):
        state = {'state_dict': self.net.state_dict(),
                 'init_state_dict': self.init_state_dict,
                 'params_l2_norm': self.params_l2_norm_list[-1],
                 'train_loss': self.train_epoch_loss_list[-1],
                 'train_acc': self.train_epoch_acc_list[-1],
                 'val_loss': self.val_epoch_loss_list[-1],
                 'val_acc': self.val_epoch_acc_list[-1]
                 }

        torch.save(state, filename)


def train_model(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

    model = GaGrad()
    model.load_data()
    model.run()


def main():
    train_model(0)


if __name__ == '__main__':
    main()
