import os
import numpy as np

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from block import block

from qpth.qp import QPFunction

class ReluNet(nn.Module):
    def __init__(self, nFeatures, nHidden, bn=False):
        super().__init__()
        self.bn = bn

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nFeatures)
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)

    def __call__(self, x):
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = self.fc2(x)
        return x

class OptNet(nn.Module):
    def __init__(self, nFeatures, args):
        super(OptNet, self).__init__()

        nHidden, neq, nineq = 2*nFeatures-1,0,2*nFeatures-2
        assert(neq==0)

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.M = Variable(torch.tril(torch.ones(nHidden, nHidden)).cuda())

        if args.tvInit:
            Q = 1e-8*torch.eye(nHidden)
            Q[:nFeatures,:nFeatures] = torch.eye(nFeatures)
            self.L = Parameter(torch.potrf(Q))

            D = torch.zeros(nFeatures-1, nFeatures)
            D[:nFeatures-1,:nFeatures-1] = torch.eye(nFeatures-1)
            D[:nFeatures-1,1:nFeatures] -= torch.eye(nFeatures-1)
            G_ = block((( D, -torch.eye(nFeatures-1)),
                        (-D, -torch.eye(nFeatures-1))))
            self.G = Parameter(G_)
            self.s0 = Parameter(torch.ones(2*nFeatures-2)+1e-6*torch.randn(2*nFeatures-2))
            G_pinv = (G_.t().mm(G_)+1e-5*torch.eye(nHidden)).inverse().mm(G_.t())
            self.z0 = Parameter(-G_pinv.mv(self.s0.data)+1e-6*torch.randn(nHidden))

            lam = 21.21
            W_fc1, b_fc1 = self.fc1.weight, self.fc1.bias
            W_fc1.data[:,:] = 1e-3*torch.randn((2*nFeatures-1, nFeatures))
            # W_fc1.data[:,:] = 0.0
            W_fc1.data[:nFeatures,:nFeatures] += -torch.eye(nFeatures)
            # b_fc1.data[:] = torch.zeros(2*nFeatures-1)
            b_fc1.data[:] = 0.0
            b_fc1.data[nFeatures:2*nFeatures-1] = lam
        else:
            self.L = Parameter(torch.tril(torch.rand(nHidden, nHidden)))
            self.G = Parameter(torch.Tensor(nineq,nHidden).uniform_(-1,1))
            self.z0 = Parameter(torch.zeros(nHidden))
            self.s0 = Parameter(torch.ones(nineq))

        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.neq = neq
        self.nineq = nineq
        self.args = args

    def cuda(self):
        # TODO: Is there a more automatic way?
        for x in [self.L, self.G, self.z0, self.s0]:
            x.data = x.data.cuda()

        return super().cuda()

    def forward(self, x):
        nBatch = x.size(0)

        x = self.fc1(x)

        L = self.M*self.L
        Q = L.mm(L.t()) + self.args.eps*Variable(torch.eye(self.nHidden)).cuda()
        Q = Q.unsqueeze(0).expand(nBatch, self.nHidden, self.nHidden)
        G = self.G.unsqueeze(0).expand(nBatch, self.nineq, self.nHidden)
        h = self.G.mv(self.z0)+self.s0
        h = h.unsqueeze(0).expand(nBatch, self.nineq)
        e = Variable(torch.Tensor())
        x = QPFunction()(Q, x, G, h, e, e)
        x = x[:,:self.nFeatures]

        return x

class OptNet_LearnD(nn.Module):
    def __init__(self, nFeatures, args):
        super().__init__()

        nHidden, neq, nineq = 2*nFeatures-1,0,2*nFeatures-2
        assert(neq==0)

        # self.fc1 = nn.Linear(nFeatures, nHidden)
        self.M = Variable(torch.tril(torch.ones(nHidden, nHidden)).cuda())

        Q = 1e-8*torch.eye(nHidden)
        Q[:nFeatures,:nFeatures] = torch.eye(nFeatures)
        self.L = Variable(torch.potrf(Q))

        self.D = Parameter(0.3*torch.randn(nFeatures-1, nFeatures))
        # self.lam = Parameter(20.*torch.ones(1))
        self.h = Variable(torch.zeros(nineq))

        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.neq = neq
        self.nineq = nineq
        self.args = args

    def cuda(self):
        # TODO: Is there a more automatic way?
        for x in [self.L, self.D, self.h]:
            x.data = x.data.cuda()

        return super().cuda()

    def forward(self, x):
        nBatch = x.size(0)

        L = self.M*self.L
        Q = L.mm(L.t()) + self.args.eps*Variable(torch.eye(self.nHidden)).cuda()
        Q = Q.unsqueeze(0).expand(nBatch, self.nHidden, self.nHidden)
        nI = Variable(-torch.eye(self.nFeatures-1).type_as(Q.data))
        G = torch.cat((
              torch.cat(( self.D, nI), 1),
              torch.cat((-self.D, nI), 1)
        ))
        G = G.unsqueeze(0).expand(nBatch, self.nineq, self.nHidden)
        h = self.h.unsqueeze(0).expand(nBatch, self.nineq)
        e = Variable(torch.Tensor())
        # p = torch.cat((-x, self.lam.unsqueeze(0).expand(nBatch, self.nFeatures-1)), 1)
        p = torch.cat((-x, Parameter(13.*torch.ones(nBatch, self.nFeatures-1).cuda())), 1)
        x = QPFunction()(Q.double(), p.double(), G.double(), h.double(), e, e).float()
        x = x[:,:self.nFeatures]

        return x
