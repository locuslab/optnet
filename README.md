# OptNet: Differentiable Optimization as a Layer in Neural Networks

This repository is by [Brandon Amos](http://bamos.github.io)
and [J. Zico Kolter](http://zicokolter.com)
and contains the [PyTorch](https://pytorch.org) source code to
reproduce the experiments in our paper
[OptNet: Differentiable Optimization as a Layer in Neural Networks](https://arxiv.org/abs/1703.00443).

If you find this repository helpful in your publications,
please consider citing our paper.

```
@article{amos2017optnet,
  title={OptNet: Differentiable Optimization as a Layer in Neural Networks},
  author={Brandon Amos and J. Zico Kolter},
  journal={arXiv preprint arXiv:1703.00443},
  year={2017}
}
```

# Informal Introduction

[Mathematical optimization](https://en.wikipedia.org/wiki/Mathematical_optimization)
is a well-studied language of expressing solutions to many real-life problems
that come up in machine learning and many other fields such as mechanics,
economics, EE, operations research, control engineering, geophysics,
and molecular modeling.
As we build our machine learning systems to interact with real
data from these fields, we often **cannot** (but sometimes can)
simply ``learn away'' the optimization sub-problems by adding more
layers in our network. Well-defined optimization problems may be added
if you have a thorough understanding of your feature space, but
oftentimes we **don't** have this understanding and resort to
automatic feature learning for our tasks.

Until this repository, **no** modern deep learning library has provided
a way of adding a learnable optimization layer (other than simply unrolling
an optimization procedure, which is inefficient and inexact) into
our model formulation that we can quickly try to see if it's a nice way
of expressing our data.

See our paper
[OptNet: Differentiable Optimization as a Layer in Neural Networks](https://arxiv.org/abs/1703.00443)
and code at
[locuslab/optnet](https://github.com/locuslab/optnet)
if you are interested in learning more about our initial exploration
in this space of automatically learning quadratic program layers
for signal denoising and sudoku.

## Setup and Dependencies

+ Python/numpy
+ [PyTorch](https://pytorch.org)
  + The code currently requires a source install from the master branch from
    [our fork](https://github.com/locuslab/pytorch) for new batch triangular
    factorization functions we have added.
    We are currently working with the PyTorch team to get these new features
    merged into Torch proper.
+ [qpth](https://github.com/locuslab/qpth):
  *Our fast QP solver for PyTorch released in conjunction with this paper.*
+ [bamos/block](https://github.com/bamos/block):
  *Our intelligent block matrix library for numpy, PyTorch, and beyond.*
+ Optional: [bamos/setGPU](https://github.com/bamos/setGPU):
  A small library to set `CUDA_VISIBLE_DEVICES` on multi-GPU systems.

# Denoising Experiments

+ The `arxiv.v1.denoising` dataset used in our experiments is available
  [here](http://joule.isr.cs.cmu.edu:11235/optnet/arxiv.v1.denoising.tgz).
  This dataset may change in future
  versions of our paper and we will add updated info here.

```
denoising
├── create.py - Script to create the denoising dataset.
├── plot.py - Plot the results from any experiment.
├── main.py - Run the FC baseline and OptNet denoising experiments. (See arguments.)
├── main.tv.py - Run the TV baseline denoising experiment.
└── run-exps.sh - Run all experiments. (May need to uncomment some lines.)
```

# Sudoku Experiments

+ The `arxiv.v1.sudoku` dataset used in our experiments is available
  [here](http://joule.isr.cs.cmu.edu:11235/optnet/arxiv.v1.sudoku.tgz).
  This dataset may change in future
  versions of our paper and we will add updated info here.

```
sudoku
├── create.py - Script to create the dataset.
├── plot.py - Plot the results from any experiment.
├── main.py - Run the FC baseline and OptNet Sudoku experiments. (See arguments.)
└── models.py - Models used for Sudoku.
```

# Classification Experiments

```
cls
├── train.py - Run the FC baseline and OptNet classification experiments. (See arguments.)
├── plot.py - Plot the results from any experiment.
└── models.py - Models used for classification.
```

# Acknowledgments

The rapid development of this work would not have been possible without
the immense amount of help from the [PyTorch](https://pytorch.org) team,
particularly [Soumith Chintala](http://soumith.ch/) and
[Adam Paszke](https://github.com/apaszke).

# Licensing

Unless otherwise stated, the source code is copyright
Carnegie Mellon University and licensed under the
[Apache 2.0 License](./LICENSE).
