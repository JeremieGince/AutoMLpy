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
from src.parameter_generator import ParameterGenerator


class RandomHpSearch(ParameterGenerator):
    def __init__(self,
                 values_dict: Union[Dict[Union[int, str], List[Union[int, float]]],
                                    Dict[Union[int, str], Iterable]],
                 **kwargs):
        """
        Used to generate the hyper-parameter (hp) space, generate random trial parameter for the exploration and get the
        best set of hp of the hp space according to the current exploration.

        Parameters
        ----------
        values_dict:
            A dictionary which contained all the possible values of each hyper-parameter
            used to generate the exploring space.
        kwargs: {
                    xi (float): Exploration parameter. Must be in [0, 1]. default: 0.1.
                    Lambda (float): Default: 1.0.
                    bandwidth (float): lenght_scale of the RBF kernel used in self.gpr. Default: 1.0.
                    max_itr (int): Max iteration of the gpo. Default: 30.
                    max_seconds (int): Max seconds that the gpo can take to make its optimisation. Default: 60**2.
        }

        """
        super(RandomHpSearch, self).__init__(values_dict, **kwargs)
        self._idx_choices = list(range(self.xx.shape[0]))
        np.random.shuffle(self._idx_choices)

    def get_trial_param(self) -> Dict[Union[str, int], object]:
        """
        Returned a set of trial parameter.

        Increase the current_itr counter.
        """
        self.current_itr += 1

        idx = self._idx_choices.pop(0)

        if len(self._idx_choices) == 0:
            self._idx_choices = list(range(self.xx.shape[0]))
            np.random.shuffle(self._idx_choices)

        t_param = self.convert_subspace_to_param(self.xx[idx])
        self.write_optimization_to_html(show=False, save=True,
                                        save_dir=f"{self.default_save_dir}/temp/html_files/",
                                        save_name=f"itr{self.current_itr}")
        return t_param
