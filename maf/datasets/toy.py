import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from . import *
import util
from Datasets import RealToy
from scipy.linalg import expm

class TOY:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, name, D, N=5000, noise_std=0.0, seed=1, data_args={}, rotate=True, itanh=True, whiten=True):

        dist = RealToy(name, D, N=N, seed=seed, noise_std=0.0, itanh=itanh, whiten=whiten, ntest=1000, data_args=data_args)
        trn, val, tst, idx, dist = dist.data, dist.valid_data, dist.test_data, dist.idx, dist

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)
        self.dist = dist
        self.seed= seed
        self.idx = idx

        self.n_dims = self.trn.x.shape[1]

    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        util.plot_hist_marginals(data_split.x)
        plt.show()

    def itrans(self, data):
        
        return self.dist.itrans(data)
