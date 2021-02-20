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


class GridHpSearch(ParameterGenerator):
    def __init__(self,
                 values_dict: Union[Dict[Union[int, str], List[Union[int, float]]],
                                    Dict[Union[int, str], Iterable]],
                 **kwargs):
        super(GridHpSearch, self).__init__(values_dict, **kwargs)
        self.idx = 0

    def reset(self):
        super(GridHpSearch, self).reset()
        self.idx = 0

    def __len__(self):
        return max(self.max_itr, len(self.xx))

    def get_trial_param(self):
        self.current_itr += 1

        t_param = self.convert_subspace_to_param(self.xx[self.idx])
        self.idx = self.current_itr % len(self)
        return t_param
