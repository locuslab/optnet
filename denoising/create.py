#!/usr/bin/env python3

import argparse
import numpy as np
import numpy.random as npr
import torch

import os, sys
import shutil

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minBps', type=int, default=1)
    parser.add_argument('--maxBps', type=int, default=10)
    parser.add_argument('--seqLen', type=int, default=100)
    parser.add_argument('--minHeight', type=int, default=10)
    parser.add_argument('--maxHeight', type=int, default=100)
    parser.add_argument('--noise', type=float, default=10)
    parser.add_argument('--nSamples', type=int, default=10000)
    parser.add_argument('--save', type=str, default='data/synthetic')
    args = parser.parse_args()

    npr.seed(0)

    save = args.save
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save)

    X, Y = [], []
    for i in range(args.nSamples):
        Xi, Yi = sample(args)
        X.append(Xi); Y.append(Yi)
        if i == 0:
            fig, ax = plt.subplots(1, 1)
            plt.plot(Xi, label='Corrupted')
            plt.plot(Yi, label='Original')
            plt.legend()
            f = os.path.join(args.save, "example.png")
            fig.savefig(f)
            print("Created {}".format(f))

    X = np.array(X)
    Y = np.array(Y)

    for loc,arr in (('features.pt', X), ('labels.pt', Y)):
        fname = os.path.join(args.save, loc)
        with open(fname, 'wb') as f:
            torch.save(torch.Tensor(arr), f)
        print("Created {}".format(fname))

def sample(args):
    nBps = npr.randint(args.minBps, args.maxBps)
    bpLocs = [0] + sorted(npr.choice(args.seqLen-2, nBps-1, replace=False)+1) + [args.seqLen]
    bpDiffs = np.diff(bpLocs)
    heights = npr.randint(args.minHeight, args.maxHeight, nBps)
    Y = []
    for d, h in zip(bpDiffs, heights):
        Y += [h]*d
    Y = np.array(Y, dtype=np.float)

    X = Y + npr.normal(0, args.noise, (args.seqLen))
    return X, Y

if __name__=='__main__':
    main()
