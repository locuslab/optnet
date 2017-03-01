#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
from tqdm import tqdm

import setGPU
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import numpy.random as npr

import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import setproctitle

import models

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def print_header(msg):
    print('===>', msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--boardSz', type=int, default=2)
    parser.add_argument('--batchSz', type=int, default=150)
    parser.add_argument('--testBatchSz', type=int, default=200)
    parser.add_argument('--nEpoch', type=int, default=100)
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--save', type=str)
    parser.add_argument('--work', type=str, default='work')
    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True
    fcP = subparsers.add_parser('fc')
    fcP.add_argument('--nHidden', type=int, nargs='+', default=[100,100])
    fcP.add_argument('--bn', action='store_true')
    convP = subparsers.add_parser('conv')
    convP.add_argument('--nHidden', type=int, default=50)
    convP.add_argument('--bn', action='store_true')
    optnetP = subparsers.add_parser('optnet')
    # optnetP.add_argument('--nHidden', type=int, default=50)
    # optnetP.add_argument('--nineq', type=int, default=100)
    optnetP.add_argument('--Qpenalty', type=float, default=0.1)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    t = '{}.{}'.format(args.boardSz, args.model)
    if args.model == 'optnet':
        t += '.Qpenalty={}'.format(args.Qpenalty)
    elif args.model == 'fc':
        t += '.nHidden:{}'.format(','.join([str(x) for x in args.nHidden]))
        if args.bn:
            t += '.bn'
    if args.save is None:
        args.save = os.path.join(args.work, t)
    setproctitle.setproctitle('bamos.sudoku.' + t)

    with open('data/{}/features.pt'.format(args.boardSz), 'rb') as f:
        X = torch.load(f)
    with open('data/{}/labels.pt'.format(args.boardSz), 'rb') as f:
        Y = torch.load(f)

    N, nFeatures = X.size(0), int(np.prod(X.size()[1:]))

    nTrain = int(N*(1.-args.testPct))
    nTest = N-nTrain

    trainX = X[:nTrain]
    trainY = Y[:nTrain]
    testX = X[nTrain:]
    testY = Y[nTrain:]

    assert(nTrain % args.batchSz == 0)
    assert(nTest % args.testBatchSz == 0)

    save = args.save
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save)

    npr.seed(1)

    print_header('Building model')
    if args.model == 'fc':
        # nHidden = 2*nFeatures-1
        nHidden = args.nHidden
        model = models.FC(nFeatures, nHidden, args.bn)
    elif args.model == 'optnet':
        model = models.OptNet(args.boardSz, args.Qpenalty)
    else:
        assert(False)

    if args.cuda:
        model = model.cuda()

    fields = ['epoch', 'loss', 'err']
    trainF = open(os.path.join(save, 'train.csv'), 'w')
    trainW = csv.writer(trainF)
    trainW.writerow(fields)
    trainF.flush()
    fields = ['epoch', 'loss', 'err']
    testF = open(os.path.join(save, 'test.csv'), 'w')
    testW = csv.writer(testF)
    testW.writerow(fields)
    testF.flush()


    if args.model == 'optnet':
        # if args.tvInit: lr = 1e-4
        # elif args.learnD: lr = 1e-2
        # else: lr = 1e-3
        lr = 1e-1
    else:
        lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writeParams(args, model, 'init')
    test(args, 0, model, testF, testW, testX, testY)
    for epoch in range(1, args.nEpoch+1):
        # update_lr(optimizer, epoch)
        train(args, epoch, model, trainF, trainW, trainX, trainY, optimizer)
        test(args, epoch, model, testF, testW, testX, testY)
        torch.save(model, os.path.join(args.save, 'latest.pth'))
        writeParams(args, model, 'latest')
        os.system('./plot.py "{}" &'.format(args.save))

def writeParams(args, model, tag):
    if args.model == 'optnet':
        A = model.A.data.cpu().numpy()
        np.savetxt(os.path.join(args.save, 'A.{}'.format(tag)), A)

def train(args, epoch, model, trainF, trainW, trainX, trainY, optimizer):
    batchSz = args.batchSz

    batch_data_t = torch.FloatTensor(batchSz, trainX.size(1), trainX.size(2), trainX.size(3))
    batch_targets_t = torch.FloatTensor(batchSz, trainY.size(1), trainX.size(2), trainX.size(3))
    if args.cuda:
        batch_data_t = batch_data_t.cuda()
        batch_targets_t = batch_targets_t.cuda()
    batch_data = Variable(batch_data_t, requires_grad=False)
    batch_targets = Variable(batch_targets_t, requires_grad=False)
    for i in range(0, trainX.size(0), batchSz):
        batch_data.data[:] = trainX[i:i+batchSz]
        batch_targets.data[:] = trainY[i:i+batchSz]
        # Fixed batch size for debugging:
        # batch_data.data[:] = trainX[:batchSz]
        # batch_targets.data[:] = trainY[:batchSz]

        optimizer.zero_grad()
        preds = model(batch_data)
        loss = nn.MSELoss()(preds, batch_targets)
        loss.backward()
        optimizer.step()

        err = get_nErr(preds.data, batch_targets.data)/batchSz
        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} Err: {:.4f}'.format(
            epoch, i+batchSz, trainX.size(0),
            float(i+batchSz)/trainX.size(0)*100,
            loss.data[0], err))

        trainW.writerow((epoch-1+float(i+batchSz)/trainX.size(0), loss.data[0], err))
        trainF.flush()

def test(args, epoch, model, testF, testW, testX, testY):
    batchSz = args.testBatchSz

    test_loss = 0
    batch_data_t = torch.FloatTensor(batchSz, testX.size(1), testX.size(2), testX.size(3))
    batch_targets_t = torch.FloatTensor(batchSz, testY.size(1), testX.size(2), testX.size(3))
    if args.cuda:
        batch_data_t = batch_data_t.cuda()
        batch_targets_t = batch_targets_t.cuda()
    batch_data = Variable(batch_data_t, volatile=True)
    batch_targets = Variable(batch_targets_t, volatile=True)

    nErr = 0
    for i in range(0, testX.size(0), batchSz):
        print('Testing model: {}/{}'.format(i, testX.size(0)), end='\r')
        batch_data.data[:] = testX[i:i+batchSz]
        batch_targets.data[:] = testY[i:i+batchSz]
        output = model(batch_data)
        test_loss += nn.MSELoss()(output, batch_targets)
        nErr += get_nErr(output.data, batch_targets.data)

    nBatches = testX.size(0)/batchSz
    test_loss = test_loss.data[0]/nBatches
    test_err = nErr/testX.size(0)
    print('TEST SET RESULTS:' + ' ' * 20)
    print('Average loss: {:.4f}'.format(test_loss))
    print('Err: {:.4f}'.format(test_err))

    testW.writerow((epoch, test_loss, test_err))
    testF.flush()

def get_nErr(pred, target):
    batchSz = pred.size(0)
    I_pred = torch.max(pred, 3)[1].squeeze().view(batchSz, -1).float()
    I_targ = torch.max(target, 3)[1].squeeze().view(batchSz, -1).float()
    return (torch.norm(I_pred-I_targ, 2, 1) != 0).sum()

if __name__=='__main__':
    main()
