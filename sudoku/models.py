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


class OptNet(nn.Module):
    def __init__(self, n, Qpenalty, trueInit=False):
        super().__init__()
        nx = (n**2)**3
        self.Q = Variable(Qpenalty*torch.eye(nx).double().cuda())
        self.G = Variable(-torch.eye(nx).double().cuda())
        self.h = Variable(torch.zeros(nx).double().cuda())
        t = get_sudoku_matrix(n)
        if trueInit:
            self.A = Parameter(torch.DoubleTensor(get_sudoku_matrix(n)).cuda())
        else:
            self.A = Parameter(torch.rand(t.shape).double().cuda())
        self.b = Variable(torch.ones(self.A.size(0)).double().cuda())

    def forward(self, puzzles):
        nBatch = puzzles.size(0)

        Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1))
        p = -puzzles.view(nBatch,-1)
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        A = self.A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1))
        b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0))

        return QPFunction(verbose=False)(p.double(), Q, G, h, A, b).float().view_as(puzzles)

# if __name__=="__main__":
#     sudoku = SolveSudoku(2, 0.2)
#     puzzle = [[4, 0, 0, 0], [0,0,4,0], [0,2,0,0], [0,0,0,1]]
#     Y = Variable(torch.DoubleTensor(np.array([[np.array(np.eye(5,4,-1)[i,:]) for i in row] for row in puzzle])).cuda())
#     solution = sudoku(Y.unsqueeze(0))
#     print(solution.view(1,4,4,4))
