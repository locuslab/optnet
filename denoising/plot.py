#!/usr/bin/env python3

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import pandas as pd
import numpy as np
import math

import os
import sys
import json
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workDir', type=str)
    args = parser.parse_args()

    trainF = os.path.join(args.workDir, 'train.csv')
    testF = os.path.join(args.workDir, 'test.csv')

    trainDf = pd.read_csv(trainF, sep=',')
    testDf = pd.read_csv(testF, sep=',')

    plotLoss(trainDf, testDf, args.workDir)

    initDf = os.path.join(args.workDir, 'D.init')
    if os.path.exists(initDf):
        initD = np.loadtxt(initDf)
        latestD = np.loadtxt(os.path.join(args.workDir, 'D.latest'))
        plotD(initD, latestD, args.workDir)

def plotLoss(trainDf, testDf, workDir):
    # fig, ax = plt.subplots(1, 1, figsize=(5,2))
    fig, ax = plt.subplots(1, 1)
    # fig.tight_layout()

    trainEpoch = trainDf['epoch'].values
    trainLoss = trainDf['loss'].values

    N = len(trainEpoch) // math.ceil(trainEpoch[-1])
    trainEpoch_, trainLoss_ = rolling(N, trainEpoch, trainLoss)
    plt.plot(trainEpoch_, trainLoss_, label='Train')
    # plt.plot(trainEpoch, trainLoss, label='Train')
    if not testDf.empty:
        plt.plot(testDf['epoch'].values, testDf['loss'].values, label='Test')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.xlim(xmin=0)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
    plt.legend()
    ax.set_yscale('log')
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "loss."+ext)
        fig.savefig(f)
        print("Created {}".format(f))

def plotD(initD, latestD, workDir):
    def p(D, fname):
        plt.clf()
        lim = max(np.abs(np.min(D)), np.abs(np.max(D)))
        clim = (-lim, lim)
        plt.imshow(D, cmap='bwr', interpolation='nearest', clim=clim)
        plt.colorbar()
        plt.savefig(os.path.join(workDir, fname))

    p(initD, 'initD.png')
    p(latestD, 'latestD.png')

    latestDs = latestD**6
    latestDs = latestDs/np.sum(latestDs, axis=1)[:,None]
    I = np.argsort(latestDs.dot(np.arange(latestDs.shape[1])))
    latestDs = latestD[I]
    initDs = initD[I]

    p(initDs, 'initD_sorted.png')
    p(latestDs, 'latestD_sorted.png')

    # Dcombined = np.concatenate((initDs, np.zeros((initD.shape[0], 10)), latestDs), axis=1)
    # p(Dcombined, 'Dcombined.png')

def rolling(N, i, loss):
    i_ = i[N-1:]
    K = np.full(N, 1./N)
    loss_ = np.convolve(loss, K, 'valid')
    return i_, loss_

if __name__ == '__main__':
    main()
