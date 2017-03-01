#!/usr/bin/env python3

import numpy as np
import numpy.random as npr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

for i in range(5):
    nz, neq, nineq = 2,0,10
    G = npr.uniform(-1., 1., (nineq,nz))
    z0 = np.zeros(nz)
    s0 = np.ones(nineq)

    l, u = -4, 4
    b = np.linspace(l, u, num=1000)
    C, D = np.meshgrid(b, b)
    Z = []
    for c,d in zip(C.ravel(), D.ravel()):
        x = np.array([c,d])
        z = np.all(G.dot(x) <= G.dot(z0)+s0).astype(np.float32)
        Z.append(z)
    Z = np.array(Z).reshape(C.shape)

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    plt.axis([l, u, l, u])
    CS = plt.contourf(C, D, Z, cmap=plt.cm.Blues)
    f = 'data/2016-11-02/init.{}.png'.format(i)
    plt.savefig(f)
    print('created '+f)
