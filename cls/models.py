import torch

import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from qpth.qp import QPFunction, QPSolvers

class Lenet(nn.Module):
    def __init__(self, nHidden, nCls=10, proj='softmax'):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(50*4*4, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)

        self.proj = proj
        self.nCls = nCls

        if proj == 'simproj':
            self.Q = Variable(0.5*torch.eye(nCls).double().cuda())
            self.G = Variable(-torch.eye(nCls).double().cuda())
            self.h = Variable(-1e-5*torch.ones(nCls).double().cuda())
            self.A = Variable((torch.ones(1, nCls)).double().cuda())
            self.b = Variable(torch.Tensor([1.]).double().cuda())
            def projF(x):
                nBatch = x.size(0)
                Q = self.Q.unsqueeze(0).expand(nBatch, nCls, nCls)
                G = self.G.unsqueeze(0).expand(nBatch, nCls, nCls)
                h = self.h.unsqueeze(0).expand(nBatch, nCls)
                A = self.A.unsqueeze(0).expand(nBatch, 1, nCls)
                b = self.b.unsqueeze(0).expand(nBatch, 1)
                x = QPFunction()(Q, -x.double(), G, h, A, b).float()
                x = x.log()
                return x
            self.projF = projF
        else:
            self.projF = F.log_softmax

    def forward(self, x):
        nBatch = x.size(0)

        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.projF(x)

class LenetOptNet(nn.Module):
    def __init__(self, nHidden=50, nineq=200, neq=0, eps=1e-4):
        super(LenetOptNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        self.qp_o = nn.Linear(50*4*4, nHidden)
        self.qp_z0 = nn.Linear(50*4*4, nHidden)
        self.qp_s0 = nn.Linear(50*4*4, nineq)

        assert(neq==0)
        self.M = Variable(torch.tril(torch.ones(nHidden, nHidden)).cuda())
        self.L = Parameter(torch.tril(torch.rand(nHidden, nHidden).cuda()))
        self.G = Parameter(torch.Tensor(nineq,nHidden).uniform_(-1,1).cuda())
        # self.z0 = Parameter(torch.zeros(nHidden).cuda())
        # self.s0 = Parameter(torch.ones(nineq).cuda())

        self.nHidden = nHidden
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

    def forward(self, x):
        nBatch = x.size(0)

        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(nBatch, -1)

        L = self.M*self.L
        Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nHidden)).cuda()
        Q = Q.unsqueeze(0).expand(nBatch, self.nHidden, self.nHidden)
        G = self.G.unsqueeze(0).expand(nBatch, self.nineq, self.nHidden)
        z0 = self.qp_z0(x)
        s0 = self.qp_s0(x)
        h = z0.mm(self.G.t())+s0
        e = Variable(torch.Tensor())
        inputs = self.qp_o(x)
        x = QPFunction()(Q, inputs, G, h, e, e)
        x = x[:,:10]

        return F.log_softmax(x)

class FC(nn.Module):
    def __init__(self, nHidden, bn):
        super().__init__()
        self.bn = bn

        self.fc1 = nn.Linear(784, nHidden)
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(nHidden, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        nBatch = x.size(0)

        # FC-ReLU-(BN)-FC-ReLU-(BN)-FC-Softmax
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)
        x = self.fc3(x)
        return F.log_softmax(x)

class OptNet(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, bn, nineq=200, neq=0, eps=1e-4):
        super().__init__()

        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.bn = bn
        self.nCls = nCls

        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(nCls)

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)

        # self.qp_z0 = nn.Linear(nCls, nCls)
        # self.qp_s0 = nn.Linear(nCls, nineq)

        assert(neq==0)
        self.M = Variable(torch.tril(torch.ones(nCls, nCls)).cuda())
        self.L = Parameter(torch.tril(torch.rand(nCls, nCls).cuda()))
        self.G = Parameter(torch.Tensor(nineq,nCls).uniform_(-1,1).cuda())
        self.z0 = Parameter(torch.zeros(nCls).cuda())
        self.s0 = Parameter(torch.ones(nineq).cuda())

        self.nineq = nineq
        self.neq = neq
        self.eps = eps

    def forward(self, x):
        nBatch = x.size(0)

        # FC-ReLU-(BN)-FC-ReLU-(BN)-QP-Softmax
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)

        L = self.M*self.L
        Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nCls)).cuda()
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
        G = self.G.unsqueeze(0).expand(nBatch, self.nineq, self.nCls)
        # z0 = self.qp_z0(x)
        # s0 = self.qp_s0(x)
        z0 = self.z0.unsqueeze(0).expand(nBatch, self.nCls)
        s0 = self.s0.unsqueeze(0).expand(nBatch, self.nineq)
        h = z0.mm(self.G.t())+s0
        e = Variable(torch.Tensor())
        inputs = x
        x = QPFunction(verbose=-1)(
            Q.double(), inputs.double(), G.double(), h.double(), e, e)
        x = x.float()
        # x = x[:,:10].float()

        return F.log_softmax(x)

class OptNetEq(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, neq, Qpenalty=0.1, eps=1e-4):
        super().__init__()

        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.nCls = nCls

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)

        self.Q = Variable(Qpenalty*torch.eye(nHidden).double().cuda())
        self.G = Variable(-torch.eye(nHidden).double().cuda())
        self.h = Variable(torch.zeros(nHidden).double().cuda())
        self.A = Parameter(torch.rand(neq,nHidden).double().cuda())
        self.b = Variable(torch.ones(self.A.size(0)).double().cuda())

        self.neq = neq

    def forward(self, x):
        nBatch = x.size(0)

        # FC-ReLU-QP-FC-Softmax
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))

        Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1))
        p = -x.view(nBatch,-1)
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        A = self.A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1))
        b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0))

        x = QPFunction(verbose=False)(Q, p.double(), G, h, A, b).float()
        x = self.fc2(x)

        return F.log_softmax(x)
