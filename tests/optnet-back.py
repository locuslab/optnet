#!/usr/bin/env python3
#
# Run these tests with: nosetests -v -d test-adact-back.py
#   This will run all functions even if one throws an assertion.
#
# For debugging: ./test-adact-back.py
#   Easier to print statements.
#   This will exit after the first assertion.

import os
import sys

import torch

import numpy as np
import numpy.random as npr
import numpy.testing as npt
np.set_printoptions(precision=2)

import numdifftools as nd
import cvxpy as cp

from torch.autograd import Function, Variable

import adact
import adact_forward_ip as aip

from solver import BlockSolver as Solver

from nose.tools import with_setup, assert_almost_equal

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

ATOL=1e-2
RTOL=1e-7
verbose = True
cuda = True

def test_back():
    npr.seed(1)
    nBatch, nz, neq, nineq = 1, 10, 1, 3
    # nz, neq, nineq = 3,3,3

    L = np.tril(np.random.randn(nz,nz)) + 2.*np.eye(nz,nz)
    Q = L.dot(L.T)+1e-4*np.eye(nz)
    G = 100.*npr.randn(nineq,nz)
    A = 100.*npr.randn(neq,nz)
    z0 = 1.*npr.randn(nz)
    s0 = 100.*np.ones(nineq)
    s0[:nineq//2] = 1e-6
    # print(np.linalg.norm(L))
    # print(np.linalg.norm(G))
    # print(np.linalg.norm(A))
    # print(np.linalg.norm(z0))
    # print(np.linalg.norm(s0))

    p = npr.randn(nBatch,nz)
    # print(np.linalg.norm(p))
    truez = npr.randn(nBatch,nz)

    af = adact.AdactFunction()
    zhat_0, nu_0, lam_0 = af.forward_single_np(p[0], L, G, A, z0, s0)
    dl_dzhat_0 = zhat_0-truez[0]
    S = Solver(L, A, G, z0, s0, 1e-8)
    S.reinit(lam_0, zhat_0)
    dp_0, dL_0, dG_0, dA_0, dz0_0, ds0_0 = af.backward_single_np_solver(
        S, zhat_0, nu_0, lam_0, dl_dzhat_0, L, G, A, z0, s0)
    # zhat_1, nu_1, lam_1 = af.forward_single_np(p[1], L, G, A, z0, s0)
    # dl_dzhat_1 = zhat_1-truez[1]
    # S.reinit(lam_1, zhat_1)
    # dp_1, dL_1, dG_1, dA_1, dz0_1, ds0_1 = af.backward_single_np_solver(
    #     S, zhat_1, nu_1, lam_1, dl_dzhat_1, L, G, A, z0, s0)

    p, L, G, A, z0, s0, truez = [torch.DoubleTensor(x) for x in [p, L, G, A, z0, s0, truez]]
    Q = torch.mm(L, L.t())+0.001*torch.eye(nz).type_as(L)
    if cuda:
        p, L, Q, G, A, z0, s0, truez = [x.cuda() for x in [p, L, Q, G, A, z0, s0, truez]]
    p, L, G, A, z0, s0 = [Variable(x) for x in [p, L, G, A, z0, s0]]
    for x in [p, L, G, A, z0, s0]: x.requires_grad = True

    # Q_LU, S_LU, R = aip.pre_factor_kkt_batch(Q, G, A, nBatch)
    # b = torch.mv(A, z0) if neq > 0 else None
    # h = torch.mv(G, z0)+s0
    # zhat_b, nu_b, lam_b = aip.forward_batch(p, Q, G, A, b, h, Q_LU, S_LU, R)

    zhats = af(p, L, G, A, z0, s0)
    dl_dzhat = zhats.data - truez
    zhats.backward(dl_dzhat)
    dp, dL, dG, dA, dz0, ds0 = [x.grad.clone() for x in [p, L, G, A, z0, s0]]

if __name__=='__main__':
    test_back()
