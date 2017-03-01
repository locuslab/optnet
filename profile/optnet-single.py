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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nTrials', type=int, default=5)
    parser.add_argument('--nBatch', type=int, default=128)
    args = parser.parse_args()

    npr.seed(0)

    print('==== CPU ===\n')
    prof(args, False)

    print('\n\n==== GPU ===\n')
    prof(args, True)

def prof(args, cuda):
    print('|    nz |   neq | nineq | cvxpy | pdipm |')
    print('|-------+-------+-------+-------+-------|')
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

def prof_instance(nz, neq, nineq, nIter, cuda):
    L = np.tril(npr.uniform(0,1, (nz,nz))) + np.eye(nz,nz)
    G = npr.randn(nineq,nz)
    A = npr.randn(neq,nz)
    z0 = npr.randn(nz)
    s0 = np.ones(nineq)
    p = npr.randn(nz)

    p, L, G, A, z0, s0 = [torch.Tensor(x) for x in [p, L, G, A, z0, s0]]
    Q = torch.mm(L, L.t())+0.001*torch.eye(nz).type_as(L)
    if cuda:
        p, L, Q, G, A, z0, s0 = [x.cuda() for x in [p, L, Q, G, A, z0, s0]]

    af = adact.AdactFunction()

    start = time.time()
    # One-time cost for numpy conversion.
    p_np, L_np, G_np, A_np, z0_np, s0_np = [adact.toNp(v) for v in [p, L, G, A, z0, s0]]
    cp = time.time()-start
    for i in range(nIter):
        start = time.time()
        zhat, nu, lam = af.forward_single_np(p_np, L_np, G_np, A_np, z0_np, s0_np)
        cp += time.time()-start

    b = torch.mv(A, z0) if neq > 0 else None
    h = torch.mv(G, z0)+s0
    L_Q, L_S, R = aip.pre_factor_kkt(Q, G, A, nineq, neq)
    pdipm = []
    for i in range(nIter):
        start = time.time()
        zhat_ip, nu_ip, lam_ip = aip.forward_single(p, Q, G, A, b, h, L_Q, L_S, R)
        pdipm.append(time.time()-start)
    return cp, np.sum(pdipm)

if __name__=='__main__':
    main()
