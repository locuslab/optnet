#!/usr/bin/env python3
#
# Run these tests with: nosetests -v -d test-adact-np.py
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

import adact
import adact_forward_ip as aip

from solver import BlockSolver as Solver

from nose.tools import with_setup, assert_almost_equal

ATOL=1e-2
RTOL=1e-7

npr.seed(1)
nz, neq, nineq = 5,0,4
# nz, neq, nineq = 3,3,3

L = np.tril(np.random.randn(nz,nz)) + 2.*np.eye(nz,nz)
Q = L.dot(L.T)+1e-8*np.eye(nz)
G = 1000.*npr.randn(nineq,nz)
A = 10000.*npr.randn(neq,nz)
z0 = 1.*npr.randn(nz)
s0 = 100.*np.ones(nineq)

p = npr.randn(nz)
truez = npr.randn(nz)

af = adact.AdactFunction()

zhat, nu, lam = af.forward_single_np(p, L, G, A, z0, s0)
dl_dzhat = zhat-truez

# dp, dL, dG, dA, dz0, ds0 = af.backward_single_np(zhat, nu, lam, dl_dzhat, L, G, A, z0, s0)

S = Solver(L, A, G, z0, s0, 1e-8)
S.reinit(lam, zhat)
dp, dL, dG, dA, dz0, ds0 = af.backward_single_np_solver(S, zhat, nu, lam, dl_dzhat, L, G, A, z0, s0)

verbose = True


def test_ip_forward():
    p_t, Q_t, G_t, A_t, z0_t, s0_t = [torch.Tensor(x) for x in [p, Q, G, A, z0, s0]]
    b = torch.mv(A_t, z0_t) if neq > 0 else None
    h = torch.mv(G_t,z0_t)+s0_t
    L_Q, L_S, R = aip.pre_factor_kkt(Q_t, G_t, A_t)

    zhat_ip, nu_ip, lam_ip = aip.forward_single(p_t, Q_t, G_t, A_t, b, h, L_Q, L_S, R)
    # Unnecessary clones here because of a pytorch bug when calling numpy
    # on a tensor with a non-zero offset.
    npt.assert_allclose(zhat, zhat_ip.clone().numpy(), rtol=RTOL, atol=ATOL)
    if neq > 0:
        npt.assert_allclose(nu, nu_ip.clone().numpy(), rtol=RTOL, atol=ATOL)
    npt.assert_allclose(lam, lam_ip.clone().numpy(), rtol=RTOL, atol=ATOL)

def test_dl_dz0():
    def f(z0):
        zhat, nu, lam = af.forward_single_np(p, L, G, A, z0, s0)
        return 0.5*np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    dz0_fd = df(z0)
    if verbose:
        print('dz0_fd: ', dz0_fd)
        print('dz0: ', dz0)
    npt.assert_allclose(dz0_fd, dz0, rtol=RTOL, atol=ATOL)

def test_dl_ds0():
    def f(s0):
        zhat, nu, lam = af.forward_single_np(p, L, G, A, z0, s0)
        return 0.5*np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    ds0_fd = df(s0)
    if verbose:
        print('ds0_fd: ', ds0_fd)
        print('ds0: ', ds0)
    npt.assert_allclose(ds0_fd, ds0, rtol=RTOL, atol=ATOL)

def test_dl_dp():
    def f(p):
        zhat, nu, lam = af.forward_single_np(p, L, G, A, z0, s0)
        return 0.5*np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    dp_fd = df(p)
    if verbose:
        print('dp_fd: ', dp_fd)
        print('dp: ', dp)
    npt.assert_allclose(dp_fd, dp, rtol=RTOL, atol=ATOL)

def test_dl_dp_batch():
    def f(p):
        zhat, nu, lam = af.forward_single_np(p, L, G, A, z0, s0)
        return 0.5*np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    dp_fd = df(p)
    if verbose:
        print('dp_fd: ', dp_fd)
        print('dp: ', dp)
    npt.assert_allclose(dp_fd, dp, rtol=RTOL, atol=ATOL)

def test_dl_dA():
    def f(A):
        A = A.reshape(neq,nz)
        zhat, nu, lam = af.forward_single_np(p, L, G, A, z0, s0)
        return 0.5*np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    dA_fd = df(A.ravel()).reshape(neq, nz)
    if verbose:
        print('dA_fd[1,:]: ', dA_fd[1,:])
        print('dA[1,:]: ', dA[1,:])
    npt.assert_allclose(dA_fd, dA, rtol=RTOL, atol=ATOL)

def test_dl_dG():
    def f(G):
        G = G.reshape(nineq,nz)
        zhat, nu, lam = af.forward_single_np(p, L, G, A, z0, s0)
        return 0.5*np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    dG_fd = df(G.ravel()).reshape(nineq, nz)
    if verbose:
        print('dG_fd[1,:]: ', dG_fd[1,:])
        print('dG[1,:]: ', dG[1,:])
    npt.assert_allclose(dG_fd, dG, rtol=RTOL, atol=ATOL)

def test_dl_dL():
    def f(l0):
        L_ = np.copy(L)
        L_[:,0] = l0
        zhat, nu, lam = af.forward_single_np(p, L_, G, A, z0, s0)
        return 0.5*np.sum(np.square(zhat - truez))

    df = nd.Gradient(f)
    dL_fd = df(L[:,0])
    dl0 = np.array(dL[:,0]).ravel()
    if verbose:
        print('dL_fd: ', dL_fd)
        print('dL: ', dl0)
    npt.assert_allclose(dL_fd, dl0, rtol=RTOL, atol=ATOL)

if __name__=='__main__':
    # test_ip_forward()
    test_dl_dp()
    # test_dl_dp_batch()
    # test_dl_dz0()
    # test_dl_ds0()
    # if neq > 0:
    #     test_dl_dA()
    # test_dl_dG()
    # test_dl_dL()
