import warnings
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from typing import Dict, Union, List, Iterable, Hashable, Tuple
import pandas as pd
from multiprocessing import Pool
import time
import os
import logging
from modules.parameter_generator import ParameterGenerator


class GridParamGen(ParameterGenerator):
    def __init__(self, values_dict):
        super(GridParamGen, self).__init__(values_dict)
        xx = np.meshgrid(*[values_dict[p] for p in self._values_names])
        self.params = list(zip(*[_xx.ravel() for _xx in xx]))
        self.idx = 0

    def reset(self):
        self.idx = 0

    def __len__(self):
        return len(self.params)

    def get_trial_param(self):
        param = self.params[self.idx]
        self.idx += 1
        return {self._values_names[i]: param[i] for i in range(len(param))}