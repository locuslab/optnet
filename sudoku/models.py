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

import cvxpy as cp

class FC(nn.Module):
    def __init__(self, nFeatures, nHidden, bn=False):
        super().__init__()
        self.bn = bn

        fcs = []
        prevSz = nFeatures
        for sz in nHidden:
            fc = nn.Linear(prevSz, sz)
            prevSz = sz
            fcs.append(fc)
        for sz in list(reversed(nHidden))+[nFeatures]:
            fc = nn.Linear(prevSz, sz)
            prevSz = sz
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs)

    def __call__(self, x):
        nBatch = x.size(0)
        Nsq = x.size(1)
        in_x = x
        x = x.view(nBatch, -1)

        for fc in self.fcs:
            x = F.relu(fc(x))

        x = x.view_as(in_x)
        ex = x.exp()
        exs = ex.sum(3).expand(nBatch, Nsq, Nsq, Nsq)
        x = ex/exs

        return x

class Conv(nn.Module):
    def __init__(self, boardSz):
        super().__init__()

        self.boardSz = boardSz

        convs = []
        Nsq = boardSz**2
        prevSz = Nsq
        szs = [512]*10 + [Nsq]
        for sz in szs:
            conv = nn.Conv2d(prevSz, sz, kernel_size=3, padding=1)
            convs.append(conv)
            prevSz = sz

        self.convs = nn.ModuleList(convs)

    def __call__(self, x):
        nBatch = x.size(0)
        Nsq = x.size(1)

        for i in range(len(self.convs)-1):
            x = F.relu(self.convs[i](x))
        x = self.convs[-1](x)

        ex = x.exp()
        exs = ex.sum(3).expand(nBatch, Nsq, Nsq, Nsq)
        x = ex/exs

        return x

def get_sudoku_matrix(n):
    X = np.array([[cp.Variable(n**2) for i in range(n**2)] for j in range(n**2)])
    cons = ([x >= 0 for row in X for x in row] +
            [cp.sum_entries(x) == 1 for row in X for x in row] +
            [sum(row) == np.ones(n**2) for row in X] +
            [sum([row[i] for row in X]) == np.ones(n**2) for i in range(n**2)] +
            [sum([sum(row[i:i+n]) for row in X[j:j+n]]) == np.ones(n**2) for i in range(0,n**2,n) for j in range(0, n**2, n)])
    f = sum([cp.sum_entries(x) for row in X for x in row])
    prob = cp.Problem(cp.Minimize(f), cons)

    A = np.asarray(prob.get_problem_data(cp.ECOS)["A"].todense())
    A0 = [A[0]]
    rank = 1
    for i in range(1,A.shape[0]):
        if np.linalg.matrix_rank(A0+[A[i]], tol=1e-12) > rank:
            A0.append(A[i])
            rank += 1

    return np.array(A0)


class OptNetEq(nn.Module):
    def __init__(self, n, Qpenalty, trueInit=False):
        super().__init__()
        nx = (n**2)**3
        self.Q = Variable(Qpenalty*torch.eye(nx).double().cuda())
        self.G = Variable(-torch.eye(nx).double().cuda())
        self.h = Variable(torch.zeros(nx).double().cuda())
        if trueInit:
            self.A = Parameter(torch.DoubleTensor(get_sudoku_matrix(n)).cuda())
        else:
            t = get_sudoku_matrix(n)
            self.A = Parameter(torch.rand(t.shape).double().cuda())
        self.b = Variable(torch.ones(self.A.size(0)).double().cuda())

    def forward(self, puzzles):
        nBatch = puzzles.size(0)

        p = -puzzles.view(nBatch,-1)

        return QPFunction(verbose=False)(
            self.Q, p.double(), self.G, self.h, self.A, self.b
        ).float().view_as(puzzles)

class OptNetIneq(nn.Module):
    def __init__(self, n, Qpenalty, nineq):
        super().__init__()
        nx = (n**2)**3
        self.Q = Variable(Qpenalty*torch.eye(nx).double().cuda())
        self.G1 = Variable(-torch.eye(nx).double().cuda())
        self.h1 = Variable(torch.zeros(nx).double().cuda())
        # if trueInit:
        #     self.A = Parameter(torch.DoubleTensor(get_sudoku_matrix(n)).cuda())
        # else:
        #     # t = get_sudoku_matrix(n)
        #     # self.A = Parameter(torch.rand(t.shape).double().cuda())
        #     # import IPython, sys; IPython.embed(); sys.exit(-1)
        self.A = Parameter(torch.rand(50,nx).double().cuda())
        self.G2 = Parameter(torch.Tensor(128, nx).uniform_(-1,1).double().cuda())
        self.z2 = Parameter(torch.zeros(nx).double().cuda())
        self.s2 = Parameter(torch.ones(128).double().cuda())
        # self.b = Variable(torch.ones(self.A.size(0)).double().cuda())

    def forward(self, puzzles):
        nBatch = puzzles.size(0)

        p = -puzzles.view(nBatch,-1)

        h2 = self.G2.mv(self.z2)+self.s2
        G = torch.cat((self.G1, self.G2), 0)
        h = torch.cat((self.h1, h2), 0)
        e = Variable(torch.Tensor())

        return QPFunction(verbose=False)(
            self.Q, p.double(), G, h, e, e
        ).float().view_as(puzzles)

class OptNetLatent(nn.Module):
    def __init__(self, n, Qpenalty, nLatent, nineq, trueInit=False):
        super().__init__()
        nx = (n**2)**3
        self.fc_in = nn.Linear(nx, nLatent)
        self.Q = Variable(Qpenalty*torch.eye(nLatent).cuda())
        self.G = Parameter(torch.Tensor(nineq, nLatent).uniform_(-1,1).cuda())
        self.z = Parameter(torch.zeros(nLatent).cuda())
        self.s = Parameter(torch.ones(nineq).cuda())
        self.fc_out = nn.Linear(nLatent, nx)

    def forward(self, puzzles):
        nBatch = puzzles.size(0)

        x = puzzles.view(nBatch,-1)
        x = self.fc_in(x)

        e = Variable(torch.Tensor())

        h = self.G.mv(self.z)+self.s
        x = QPFunction(verbose=False)(
            self.Q, x, self.G, h, e, e,
        )

        x = self.fc_out(x)
        x = x.view_as(puzzles)
        return x


# if __name__=="__main__":
#     sudoku = SolveSudoku(2, 0.2)
#     puzzle = [[4, 0, 0, 0], [0,0,4,0], [0,2,0,0], [0,0,0,1]]
#     Y = Variable(torch.DoubleTensor(np.array([[np.array(np.eye(5,4,-1)[i,:]) for i in row] for row in puzzle])).cuda())
#     solution = sudoku(Y.unsqueeze(0))
#     print(solution.view(1,4,4,4))
