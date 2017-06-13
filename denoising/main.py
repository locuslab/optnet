#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
from tqdm import tqdm

try: import setGPU
except ImportError: pass

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
    parser.add_argument('--batchSz', type=int, default=150)
    parser.add_argument('--testBatchSz', type=int, default=100)
    parser.add_argument('--nEpoch', type=int, default=100)
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--work', type=str, default='work')
    parser.add_argument('--save', type=str)
    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True
    reluP = subparsers.add_parser('relu')
    reluP.add_argument('--nHidden', type=int, default=50)
    reluP.add_argument('--bn', action='store_true')
    optnetP = subparsers.add_parser('optnet')
    # optnetP.add_argument('--nHidden', type=int, default=50)
    # optnetP.add_argument('--nineq', type=int, default=100)
    optnetP.add_argument('--eps', type=float, default=1e-4)
    optnetP.add_argument('--tvInit', action='store_true')
    optnetP.add_argument('--learnD', action='store_true')
    optnetP.add_argument('--Dpenalty', type=float, default=1e-1)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.save = args.save or 'work/{}.{}'.format(args.dataset, args.model)
    if args.save is None:
        t = os.path.join(args.work, args.model)
        if args.model == 'optnet':
            t += '.eps={}'.format(args.eps)
            if args.tvInit:
                t += '.tvInit'
            if args.learnD:
                t += '.learnD.{}'.format(args.Dpenalty)
        elif args.model == 'relu':
            t += '.nHidden:{}'.format(args.nHidden)
            if args.bn:
                t += '.bn'
        args.save = t
    setproctitle.setproctitle('bamos.' + args.save)

    with open('data/synthetic/features.pt', 'rb') as f:
        X = torch.load(f)
    with open('data/synthetic/labels.pt', 'rb') as f:
        Y = torch.load(f)

    N, nFeatures = X.size()

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
    if args.model == 'relu':
        # nHidden = 2*nFeatures-1
        nHidden = args.nHidden
        model = models.ReluNet(nFeatures, nHidden, args.bn)
    elif args.model == 'optnet':
        if args.learnD:
            model = models.OptNet_LearnD(nFeatures, args)
        else:
            model = models.OptNet(nFeatures, args)

    if args.cuda:
        model = model.cuda()

    fields = ['epoch', 'loss']
    trainF = open(os.path.join(save, 'train.csv'), 'w')
    trainW = csv.writer(trainF)
    trainW.writerow(fields)
    trainF.flush()
    testF = open(os.path.join(save, 'test.csv'), 'w')
    testW = csv.writer(testF)
    testW.writerow(fields)
    testF.flush()


    if args.model == 'optnet':
        if args.tvInit: lr = 1e-4
        elif args.learnD: lr = 1e-2
        else: lr = 1e-3
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
    if args.model == 'optnet' and args.learnD:
        D = model.D.data.cpu().numpy()
        np.savetxt(os.path.join(args.save, 'D.{}'.format(tag)), D)

def train(args, epoch, model, trainF, trainW, trainX, trainY, optimizer):
    batchSz = args.batchSz

    batch_data_t = torch.FloatTensor(batchSz, trainX.size(1))
    batch_targets_t = torch.FloatTensor(batchSz, trainY.size(1))
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
        mseLoss = nn.MSELoss()(preds, batch_targets)
        if args.model == 'optnet' and args.learnD:
            loss = mseLoss + args.Dpenalty*(model.D.norm(1))
        else:
            loss = mseLoss
        loss.backward()
        optimizer.step()

        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
            epoch, i+batchSz, trainX.size(0),
            float(i+batchSz)/trainX.size(0)*100,
            mseLoss.data[0]))

        trainW.writerow((epoch-1+float(i+batchSz)/trainX.size(0), mseLoss.data[0]))
        trainF.flush()

def test(args, epoch, model, testF, testW, testX, testY):
    batchSz = args.testBatchSz

    test_loss = 0
    batch_data_t = torch.FloatTensor(batchSz, testX.size(1))
    batch_targets_t = torch.FloatTensor(batchSz, testY.size(1))
    if args.cuda:
        batch_data_t = batch_data_t.cuda()
        batch_targets_t = batch_targets_t.cuda()
    batch_data = Variable(batch_data_t, volatile=True)
    batch_targets = Variable(batch_targets_t, volatile=True)

    for i in range(0, testX.size(0), batchSz):
        print('Testing model: {}/{}'.format(i, testX.size(0)), end='\r')
        batch_data.data[:] = testX[i:i+batchSz]
        batch_targets.data[:] = testY[i:i+batchSz]
        output = model(batch_data)
        if i == 0:
            testOut = os.path.join(args.save, 'test-imgs')
            os.makedirs(testOut, exist_ok=True)
            for j in range(4):
                X = batch_data.data[j].cpu().numpy()
                Y = batch_targets.data[j].cpu().numpy()
                Yhat = output[j].data.cpu().numpy()

                fig, ax = plt.subplots(1, 1)
                plt.plot(X, label='Corrupted')
                plt.plot(Y, label='Original')
                plt.plot(Yhat, label='Predicted')
                plt.legend()
                f = os.path.join(testOut, '{}.png'.format(j))
                fig.savefig(f)
        test_loss += nn.MSELoss()(output, batch_targets)

    nBatches = testX.size(0)/batchSz
    test_loss = test_loss.data[0]/nBatches
    print('TEST SET RESULTS:' + ' ' * 20)
    print('Average loss: {:.4f}'.format(test_loss))

    testW.writerow((epoch, test_loss))
    testF.flush()

if __name__=='__main__':
    main()
