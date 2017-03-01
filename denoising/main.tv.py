#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
from tqdm import tqdm

import cvxpy as cp

import torch

import numpy as np
import numpy.random as npr

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('text', usetex=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nEpoch', type=int, default=50)
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--workDir', type=str, default='work/tv')
    args = parser.parse_args()

    with open('data/synthetic/features.pt', 'rb') as f:
        X = torch.load(f).numpy()
    with open('data/synthetic/labels.pt', 'rb') as f:
        Y = torch.load(f).numpy()

    N, nFeatures = X.shape
    nTrain = int(N*(1.-args.testPct))
    nTest = N-nTrain

    trainX = X[:nTrain]
    trainY = Y[:nTrain]
    testX = X[nTrain:]
    testY = Y[nTrain:]

    workDir = args.workDir
    if os.path.isdir(workDir):
        shutil.rmtree(workDir)
    os.makedirs(workDir)

    npr.seed(1)

    X_ = cp.Parameter(nFeatures)
    Y_ = cp.Variable(nFeatures)
    lams = list(np.linspace(0,100,101))
    mses = []

    def getMse(lam):
        prob = cp.Problem(cp.Minimize(0.5*cp.sum_squares(X_-Y_)+lam*cp.tv(Y_)))
        mses_lam = []

        # testOut = os.path.join(workDir, 'test-imgs', 'lam-{:07.2f}'.format(lam))
        # os.makedirs(testOut, exist_ok=True)

        for i in range(nTest):
            X_.value = testX[i]
            prob.solve(cp.SCS)
            assert('optimal' in prob.status)
            Yhat = np.array(Y_.value).ravel()
            mse = np.mean(np.square(testY[i] - Yhat))

            mses_lam.append(mse)

            # if i <= 4:
            #     fig, ax = plt.subplots(1, 1)
            #     plt.plot(testX[i], label='Corrupted')
            #     plt.plot(testY[i], label='Original')
            #     plt.plot(Yhat, label='Predicted')
            #     plt.legend()
            #     f = os.path.join(testOut, '{}.png'.format(i))
            #     fig.savefig(f)
            #     plt.close(fig)

        return np.mean(mses_lam)

    for lam in lams:
        mses.append(getMse(lam))
        print(lam, mses[-1])

    xMin, xMax = (1, 30)

    with open(os.path.join(workDir, 'mses.csv'), 'w') as f:
        for lam,mse in zip(lams,mses):
            f.write('{},{}\n'.format(lam,mse))

    fig, ax = plt.subplots(1, 1)
    plt.plot(lams, mses)
    plt.xlabel("$\lambda$")
    plt.ylabel("MSE")
    # plt.xlim(xmin=0)
    # ax.set_yscale('log')
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "loss."+ext)
        fig.savefig(f)
        print("Created {}".format(f))

if __name__=='__main__':
    main()
