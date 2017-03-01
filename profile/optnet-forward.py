#!/usr/bin/env python3

import argparse
import sys

import numpy as np
import numpy.random as npr

import adact
import adact_forward_ip as aip

import itertools
import time

import torch

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nTrials', type=int, default=5)
    parser.add_argument('--nBatch', type=int, default=128)
    args = parser.parse_args()

    npr.seed(0)

    # print('==== CPU ===\n')
    # prof(args, False)

    print('\n\n==== GPU ===\n')
    prof(args, True)

def prof(args, cuda):
    print('|    nz |   neq | nineq | single | batched |')
    print('|-------+-------+-------+--------+---------|')
    for nz,neq,nineq in itertools.product([10,100,200], [0,10,50], [10,50]):
        if nz >= neq and nz >= nineq:
            times = []
            for i in range(args.nTrials):
                times.append(prof_instance(nz, neq, nineq, args.nBatch, cuda))
            times = np.array(times)
            cp, pdipm = times.mean(axis=0)
            cp_sd, pdipm_sd = times.std(axis=0)
            print("| {:5d} | {:5d} | {:5d} | {:.3f} +/- {:.3f} | {:.3f} +/- {:.3f} |".format(
                nz, neq, nineq, cp, cp_sd, pdipm, pdipm_sd))

def prof_instance(nz, neq, nineq, nBatch, cuda):
    L = np.tril(npr.uniform(0,1, (nz,nz))) + np.eye(nz,nz)
    G = npr.randn(nineq,nz)
    A = npr.randn(neq,nz)
    z0 = npr.randn(nz)
    s0 = np.ones(nineq)
    p = npr.randn(nBatch,nz)

    p, L, G, A, z0, s0 = [torch.Tensor(x) for x in [p, L, G, A, z0, s0]]
    Q = torch.mm(L, L.t())+0.001*torch.eye(nz).type_as(L)
    if cuda:
        p, L, Q, G, A, z0, s0 = [x.cuda() for x in [p, L, Q, G, A, z0, s0]]
    b = torch.mv(A, z0) if neq > 0 else None
    h = torch.mv(G, z0)+s0

    af = adact.AdactFunction()

    single_results = []
    start = time.time()
    U_Q, U_S, R = aip.pre_factor_kkt(Q, G, A)
    for i in range(nBatch):
        single_results.append(aip.forward_single(p[i], Q, G, A, b, h, U_Q, U_S, R))
    single_time = time.time()-start

    start = time.time()
    Q_LU, S_LU, R = aip.pre_factor_kkt_batch(Q, G, A, nBatch)
    zhat_b, nu_b, lam_b = aip.forward_batch(p, Q, G, A, b, h, Q_LU, S_LU, R)
    batched_time = time.time()-start

    zhat_diff = (single_results[0][0] - zhat_b[0]).norm()
    lam_diff = (single_results[0][2] - lam_b[0]).norm()
    eps = 0.1 # Pretty relaxed.
    if zhat_diff > eps or lam_diff > eps:
        print('===========')
        print("Warning: Single and batched solutions might not match.")
        print("  + zhat_diff: {}".format(zhat_diff))
        print("  + lam_diff: {}".format(lam_diff))
        print("  + (nz, neq, nineq, nBatch) = ({}, {}, {}, {})".format(
            nz, neq, nineq, nBatch))
        print('===========')

    return single_time, batched_time

if __name__=='__main__':
    main()
