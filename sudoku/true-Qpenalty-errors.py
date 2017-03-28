#!/usr/bin/env python3

import torch
from torch.autograd import Variable

import numpy as np

import models
from train import computeErr

for boardSz in (2,3):
    print('# Board Sz: {}'.format(boardSz))
    print('| {:15s} | {:15s} | {:15s} |'.format('Qpenalty', '% Boards Wrong', 'MSE'))
    with open('data/{}/features.pt'.format(boardSz), 'rb') as f:
        unsolvedBoards = Variable(torch.load(f).cuda()[:,:,:,:])
        nBoards = unsolvedBoards.size(0)
    with open('data/{}/labels.pt'.format(boardSz), 'rb') as f:
        solvedBoards = Variable(torch.load(f).cuda()[:nBoards,:,:,:])
    for Qpenalty in np.logspace(-3, 1, num=10, base=10.):
        model = models.OptNetEq(boardSz, Qpenalty, trueInit=True).cuda()
        preds = model(unsolvedBoards).data
        err = computeErr(preds)/nBoards
        # MSE is not an exact metric because a board might have multiple solutions.
        mse = (preds - solvedBoards.data).mean()
        print('| {:15f} | {:15f} | {:15e} |'.format(Qpenalty, err, mse))
    print('\n\n')
