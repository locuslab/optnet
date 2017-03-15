#!/usr/bin/env python3

import torch
from torch.autograd import Variable

import numpy as np

import models
from train import computeErr

for boardSz in (2,3):
    print('# Board Sz: {}'.format(boardSz))
    print('| {:15s} | {:15s} |'.format('Qpenalty', 'Error'))
    with open('data/{}/features.pt'.format(boardSz), 'rb') as f:
        unsolvedBoards = Variable(torch.load(f).cuda()[:5,:,:,:])
        nBoards = unsolvedBoards.size(0)
    for Qpenalty in np.logspace(-3, 1, num=10, base=10.):
        try:
            model = models.OptNet(boardSz, Qpenalty, trueInit=True).cuda()
            err = computeErr(model(unsolvedBoards).data)/nBoards
            print('| {:15f} | {:15f} |'.format(Qpenalty, err))
        except:
            continue
    print('\n\n')
