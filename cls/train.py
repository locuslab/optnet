#!/usr/bin/env python3

import json

import argparse

import setGPU
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Function, Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math

import shutil

import setproctitle

import densenet
import models
# import make_graph

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def get_loaders(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        trainLoader = torch.utils.data.DataLoader(
            dset.MNIST('data/mnist', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batchSz, shuffle=True, **kwargs)
        testLoader = torch.utils.data.DataLoader(
            dset.MNIST('data/mnist', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batchSz, shuffle=False, **kwargs)
    elif args.dataset == 'cifar-10':
        normMean = [0.49139968, 0.48215827, 0.44653124]
        normStd = [0.24703233, 0.24348505, 0.26158768]
        normTransform = transforms.Normalize(normMean, normStd)

        trainTransform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normTransform
        ])
        testTransform = transforms.Compose([
            transforms.ToTensor(),
            normTransform
        ])

        trainLoader = DataLoader(
            dset.CIFAR10(root='data/cifar', train=True, download=True,
                        transform=trainTransform),
            batch_size=args.batchSz, shuffle=True, **kwargs)
        testLoader = DataLoader(
            dset.CIFAR10(root='data/cifar', train=False, download=True,
                        transform=testTransform),
            batch_size=args.batchSz, shuffle=False, **kwargs)
    else:
        assert(False)

    return trainLoader, testLoader

def get_net(args):
    if args.model == 'densenet':
        net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                                bottleneck=True, nClasses=10)
    elif args.model == 'lenet':
        net = models.Lenet(args.nHidden, 10, args.proj)
    elif args.model == 'lenet-optnet':
        net = models.LenetOptNet(args.nHidden, args.nineq)
    elif args.model == 'fc':
        net = models.FC(args.nHidden, args.bn)
    elif args.model == 'optnet':
        net = models.OptNet(28*28, args.nHidden, 10, args.bn, args.nineq)
    elif args.model == 'optnet-eq':
        net = models.OptNetEq(28*28, args.nHidden, 10, args.neq)
    else:
        assert(False)

    return net

def get_optimizer(args, params):
    if args.dataset == 'mnist':
        if args.model == 'optnet-eq':
            params = list(params)
            A_param = params.pop(0)
            assert(A_param.size() == (args.neq, args.nHidden))
            optimizer = optim.Adam([
                {'params': params, 'lr': 1e-3},
                {'params': [A_param], 'lr': 1e-1}
            ])
        else:
            optimizer = optim.Adam(params)
    elif args.dataset in ('cifar-10', 'cifar-100'):
        if args.opt == 'sgd':
            optimizer = optim.SGD(params, lr=1e-1, momentum=0.9, weight_decay=args.weightDecay)
        elif args.opt == 'adam':
            optimizer = optim.Adam(params, weight_decay=args.weightDecay)
    else:
        assert(False)

    return optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save', type=str)
    parser.add_argument('--work', type=str, default='work')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--nEpoch', type=int, default=1000)
    parser.add_argument('--weightDecay', type=float, default=1e-4)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam'))
    parser.add_argument('dataset', type=str,
                        choices=['mnist', 'cifar-10', 'cifar-100', 'svhn'])
    subparsers = parser.add_subparsers(dest='model')
    lenetP = subparsers.add_parser('lenet')
    lenetP.add_argument('--nHidden', type=int, default=50)
    lenetP.add_argument('--proj', type=str, choices=('softmax', 'simproj'))
    lenetOptnetP = subparsers.add_parser('lenet-optnet')
    lenetOptnetP.add_argument('--nHidden', type=int, default=50)
    lenetOptnetP.add_argument('--nineq', type=int, default=100)
    lenetOptnetP.add_argument('--eps', type=float, default=1e-4)
    densenetP = subparsers.add_parser('densenet')
    fcP = subparsers.add_parser('fc')
    fcP.add_argument('--nHidden', type=int, default=500)
    fcP.add_argument('--bn', action='store_true')
    optnetP = subparsers.add_parser('optnet')
    optnetP.add_argument('--nHidden', type=int, default=500)
    optnetP.add_argument('--eps', default=1e-4)
    optnetP.add_argument('--nineq', type=int, default=10)
    optnetP.add_argument('--bn', action='store_true')
    optnetEqP = subparsers.add_parser('optnet-eq')
    optnetEqP.add_argument('--nHidden', type=int, default=100)
    optnetEqP.add_argument('--neq', type=int, default=50)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.save is None:
        t = '{}.{}'.format(args.dataset, args.model)
        if args.model == 'lenet':
            t += '.nHidden:{}.proj:{}'.format(args.nHidden, args.proj)
        elif args.model == 'fc':
            t += '.nHidden:{}'.format(args.nHidden)
            if args.bn:
                t += '.bn'
        elif args.model == 'optnet':
            t += '.nHidden:{}.nineq:{}.eps:{}'.format(args.nHidden, args.nineq, args.eps)
            if args.bn:
                t += '.bn'
        elif args.model == 'optnet-eq':
            t += '.nHidden:{}.neq:{}'.format(args.nHidden, args.neq)
        elif args.model == 'lenet-optnet':
            t += '.nHidden:{}.nineq:{}.eps:{}'.format(args.nHidden, args.nineq, args.eps)
    setproctitle.setproctitle('bamos.'+t)
    args.save = os.path.join(args.work, t)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    trainLoader, testLoader = get_loaders(args)
    net = get_net(args)
    optimizer = get_optimizer(args, net.parameters())

    args.nparams = sum([p.data.nelement() for p in net.parameters()])
    with open(os.path.join(args.save, 'meta.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=2)

    print('  + Number of params: {}'.format(args.nparams))
    if args.cuda:
        net = net.cuda()

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpoch + 1):
        adjust_opt(args, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test(args, epoch, net, testLoader, optimizer, testF)
        try:
            torch.save(net, os.path.join(args.save, 'latest.pth'))
        except:
            pass
        os.system('./plot.py "{}" &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

def adjust_opt(args, optimizer, epoch):
    if args.model == 'densenet':
        if args.opt == 'sgd':
            if epoch == 150: update_lr(optimizer, 1e-2)
            elif epoch == 225: update_lr(optimizer, 1e-3)
            else: return

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__=='__main__':
    main()
