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


class ParameterGenerator:
    def __init__(self,
                 values_dict: Dict[Union[int, str], Iterable[Union[int, float]]],
                 **kwargs):
        """
        Used to generate the hyper-parameter (hp) space, generate trial parameter for the exploration and get the best
        set of hp of the hp space according to the current exploration.

        Parameters
        ----------
        values_dict:
            A dictionary which contained all the possible values of each hyper-parameter
            used to generate the exploring space.
        kwargs: {

        }

        Attributes
        ----------
        self._param_name_to_idx (Dict[str, int]) : Dict container for string bounds element used to convert str to idx.
        self._param_idx_to_name (Dict[int, str]) : Dict container for string bounds element used to convert back idx
                                                   to str.
        self.current_itr (int): Current iteration of the gpo.
        self.start_time (int): Starting time of the optimisation.
        self.history List[Tuple]: The history of the hp search.
        """
        self._param_name_to_idx = {}
        self._param_idx_to_name = {}

        self._values_names = list(values_dict.keys())
        self._values_dict = values_dict

        for p in self._values_names:
            if self.check_str_in_iterable(values_dict[p]):
                self.add_conversion_tables_param_name_to_idx(p, values_dict[p])
                values_dict[p] = self.convert_param_to_idx(p, values_dict[p])

        self.xx = np.meshgrid(*[values_dict[p] for p in self._values_names])
        self.xx = np.array(list(zip(*[_x.ravel() for _x in self.xx])))

        self.current_itr: int = 0
        self.max_itr = kwargs.get("max_itr", 30)
        self.start_time = time.time()
        self.max_seconds = kwargs.get("max_seconds", 60 ** 2)
        self.history = []

    @property
    def bounds_names(self) -> List[str]:
        return self._values_names

    def reset(self) -> None:
        """
        Reset the current parameter generator.
        """
        self.start_time = time.time()
        self.current_itr = 0

    @property
    def elapse_time(self) -> float:
        return time.time() - self.start_time

    def __len__(self) -> int:
        """
        Returned the number of trial that the current param gen can make.
        """
        return self.max_itr

    def __bool__(self) -> bool:
        """
        Returned True if the current param gen can return trial parameters.
        """
        return self.current_itr < self.max_itr and self.elapse_time < self.max_seconds

    def get_trial_param(self) -> Dict[str, Union[int, float]]:
        """
        Returned a set of trial parameter.
        """
        raise NotImplementedError()

    def get_best_param(self) -> Dict[str, Union[int, float]]:
        """
        Get the best predicted parameters with the current exploration.
        """
        return max(self.history, key=lambda t: t[-1])[0]

    def add_score_info(self, param: Dict[str, Union[int, float]], score: float) -> None:
        """
        Add the result of the trial parameters.

        Parameters
        ----------
        param: The trial parameters.
        score: The associated score of the trial parameters.
        """
        self.history.append((param, score))
        for p_name in param:
            if p_name in self._param_name_to_idx:
                param[p_name] = self._param_name_to_idx[p_name][param[p_name]]

    def add_conversion_tables_param_name_to_idx(self, param_name: str, values: Iterable) -> Dict[Hashable, int]:
        """
        Add a parameter to the conversion tables.

        Parameters
        ----------
        param_name: The parameter name to show its expectation.
        values: The values of the parameter.

        Return
        ---------
        The conversion dict self._param_name_to_idx for the given parameter.
        """
        self._param_name_to_idx[param_name] = {v: i for i, v in enumerate(values)}
        self._param_idx_to_name[param_name] = {i: v for i, v in enumerate(values)}
        return self._param_name_to_idx[param_name]

    def convert_param_to_idx(self, param_name: str, values: Iterable) -> List[int]:
        """
        Convert a parameter to idx using the conversion tables.

        Parameters
        ----------
        param_name: The parameter name to show its expectation.
        values: The values of the parameter.

        Return
        ---------
        The conversion of values as List.
        """
        assert param_name in self._param_name_to_idx
        return [self._param_name_to_idx[param_name][v] for v in values]

    def convert_idx_to_param(self, param_name: str, indexes: Iterable[int]) -> List[int]:
        """
        Convert a parameter to name using the conversion tables.

        Parameters
        ----------
        param_name: The parameter name to show its expectation.
        indexes: The indexes of the parameter.

        Return
        ---------
        The conversion of indexes as List.
        """
        assert param_name in self._param_idx_to_name
        return [self._param_idx_to_name[param_name][idx] for idx in indexes]

    def convert_subspace_to_param(self, sub_space: np.ndarray) -> dict:
        """
        Convert a subspace of space self.xx to a set of parameters.
        Parameters
        ----------
        sub_space: the subspace of self.xx. ex: self.xx[int]

        Returns
        -------
        The hyper-parameters of the subspace as a dictionary.
        """
        _param = {self._values_names[i]: sub_space[i] for i in range(len(sub_space))}
        for p_name in _param:
            if p_name in self._param_idx_to_name:
                _param[p_name] = self._param_idx_to_name[p_name][_param[p_name]]
        return _param

    @staticmethod
    def check_str_in_iterable(iterable: Iterable) -> bool:
        """
        Check if an iterable contain a str in it.

        Parameters
        ----------
        iterable: Iterable of values.

        Return
        ---------
        True if the iterable contain a str in it.
        """
        return any([isinstance(e, str) for e in iterable])

    def show_expectation(self, **kwargs) -> None:
        """
        Show the expectation of hp-space.
        """
        pass

    def save_best_param(self, **kwargs):
        save_dir = kwargs.get("save_dir", "optimal_hp/")
        save_name = kwargs.get("save_name", "opt_hp")
        save_path = save_dir + '/' + save_name + ".npy"
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_path, self.get_best_param(), allow_pickle=True)