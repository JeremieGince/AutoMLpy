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
import torch
import logging
from modules.parameter_generator import ParameterGenerator
from enum import Enum


class ParamGenType:
    Grid = 0
    Random = 1
    GPO = 2


class HpOptimizer:
    def __init__(self):
        self.hp = {}
        self.model = None

    def build_model(self, **hp) -> object:
        raise NotImplementedError("build_model method must be implemented by the user")

    def fit_model(
            self,
            model: object,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            **hp
    ) -> None:
        raise NotImplementedError("fit_model method must be implemented by the user")

    def score(
            self,
            model: object,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            **hp
    ) -> Tuple[float, float]:
        raise NotImplementedError("score method must be implemented by the user")

    def __call__(self, *args, **kwargs):
        return self.optimize(*args, **kwargs)

    def optimize(
            self,
            param_gen: ParameterGenerator,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            n_splits: int = 2,
            save_kwargs: dict = None,
            verbose: bool = True,
    ):
        """
        Optimize hyper-paremeters using the given ParameterGenerator and kfolding.

        Parameters
        ----------
        param_gen: The parameter generator.
        X: The training data.
        y: The training labels.
        n_splits: Number of split for the kfold.
        save_kwargs:
        verbose: True to print some stats else False.

        Return
        ---------
        The given ParameterGenerator optimized.
        """
        if save_kwargs is None:
            save_kwargs = {}
        warnings.simplefilter("ignore", UserWarning)

        param_gen.reset()
        if verbose:
            progress = tqdm.tqdm(range(len(param_gen)), unit='itr', postfix="optimisation")
        else:
            progress = range(len(param_gen))
        for _ in progress:
            if not bool(param_gen):
                break
            try:
                params = param_gen.get_trial_param()

                kf = KFold(n_splits=n_splits, shuffle=True)

                mean_score = 0
                for j, (train_index, test_index) in enumerate(kf.split(X)):
                    try:
                        if isinstance(X, (np.ndarray, torch.Tensor)):
                            sub_X_train, sub_X_test = X[train_index], X[test_index]
                            sub_y_train, sub_y_test = y[train_index], y[test_index]
                        elif isinstance(X, pd.DataFrame):
                            sub_X_train, sub_X_test = X.iloc[train_index], X.iloc[test_index]
                            sub_y_train, sub_y_test = y[train_index], y[test_index]
                        else:
                            raise ValueError(f"X must be Union[np.ndarray, pd.DataFrame, torch.Tensor]")

                        model = self.build_model(**params)
                        self.fit_model(model, sub_X_train, **params)

                        score, _ = self.score(model, sub_X_test, sub_y_test, **params)
                        mean_score = (j * mean_score + score) / (j + 1)
                    except Exception as e:
                        logging.error(str(e))

                if verbose:
                    progress.set_postfix_str(f"mean_score: {mean_score:.2f}")
                    progress.update()

                param_gen.add_score_info(params, mean_score)
            except Exception as e:
                logging.error(str(e))

        if verbose:
            progress.close()
        param_gen.save_best_param(**save_kwargs)
        param_gen.show_expectation(show=False, **save_kwargs)
        return param_gen


def optimize_parameters(
        model_cls,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray],
        param_gen: ParameterGenerator,
        fit_func_name: str = "fit",
        score_func_name: str = "score",
        n_splits: int = 2,
        model_cls_args: List = None,
        fit_kwargs: dict = None,
        forward_kwargs: dict = None,
        save_kwargs: dict = None,
        verbose: bool = True,
) -> ParameterGenerator:
    """
    Optimize hyper-paremeters using the given ParameterGenerator and kfolding.

    Parameters
    ----------
    model_cls: A class model. Must have the following implemented methods:
                -> __init__(*model_cls_args, **params)
    X: The training data.
    y: The training labels.
    param_gen: The parameter generator.
    fit_func_name: The name of the method function to fit the model with the signature:
                    -> fit(X: Union[np.ndarray pd.DataFrame],
                           y: Union[np.ndarray, ],
                           **fit_kwargs)
    score_func_name: The name of the method used to get the score with signature:
                     -> clf.score(X: Union[np.ndarray, pd.DataFrame],
                                  y: Union[np.ndarray, ],
                                  **forward_kwargs) -> Tuple[float, float] i.e: (mean, std)
    n_splits: Number of split for the kfold.
    model_cls_args: Arguments to give to the model_cls constructor.
    fit_kwargs: kwargs to give to the fit method of model_cls.
    forward_kwargs: kwargs to give to the score method of model_cls.
    save_kwargs:
    verbose: True to print some stats else False.

    Return
    ---------
    The given ParameterGenerator optimized.
    """
    if save_kwargs is None:
        save_kwargs = {}
    if forward_kwargs is None:
        forward_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}
    if model_cls_args is None:
        model_cls_args = []
    warnings.simplefilter("ignore", UserWarning)

    param_gen.reset()
    if verbose:
        progress = tqdm.tqdm(range(len(param_gen)), unit='itr', postfix="optimisation")
    else:
        progress = range(len(param_gen))
    for _ in progress:
        if not bool(param_gen):
            break
        try:
            params = param_gen.get_trial_param()

            kf = KFold(n_splits=n_splits, shuffle=True)

            mean_score = 0
            for j, (train_index, test_index) in enumerate(kf.split(X)):
                try:
                    if isinstance(X, (np.ndarray, )):
                        sub_X_train, sub_X_test = X[train_index], X[test_index]
                        sub_y_train, sub_y_test = y[train_index], y[test_index]
                    elif isinstance(X, pd.DataFrame):
                        sub_X_train, sub_X_test = X.iloc[train_index], X.iloc[test_index]
                        sub_y_train, sub_y_test = y[train_index], y[test_index]
                    else:
                        raise ValueError(f"X must be Union[np.ndarray, pd.DataFrame]")

                    clf = model_cls(*model_cls_args, **params)
                    getattr(clf, fit_func_name)(clf, sub_X_train, sub_y_train, **fit_kwargs)

                    score, _ = getattr(clf, score_func_name)(clf, sub_X_test, sub_y_test, **forward_kwargs)
                    mean_score = (j * mean_score + score) / (j + 1)
                except Exception as e:
                    logging.error(str(e))

            if verbose:
                progress.set_postfix_str(f"mean_score: {mean_score:.2f}")
                progress.update()

            param_gen.add_score_info(params, mean_score)
        except Exception as e:
            logging.error(str(e))

    if verbose:
        progress.close()
    param_gen.save_best_param(**save_kwargs)
    param_gen.show_expectation(show=False, **save_kwargs)
    return param_gen


def multi_gpo_precessing(
    model_cls,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, ],
    param_gens: List[ParameterGenerator],
    fit_func_name: str = "fit",
    score_func_name: str = "score",
    n_splits: int = 2,
    model_cls_args: List = None,
    fit_kwargs: dict = None,
    forward_kwargs: dict = None,
    save_kwargs_list: List = None,
    verbose: bool = True,
    optimisation_func=optimize_parameters,
    nb_cpu: Union[int, str] = "max",
) -> List[ParameterGenerator]:
    """
    Optimize hyper-paremeters using the given ParamGens and kfolding in multiprocessing.

    Parameters
    ----------
    model_cls: A class model. Must have the following implemented methods:
                -> __init__(*model_cls_args, **params)
    X: The training data.
    y: The training label.
    param_gens: List of the parameter generator.
    fit_func_name: The name of the method function to fit the model with the signature:
                    -> fit(X: Union[np.ndarray, pd.DataFrame],
                           y: Union[np.ndarray, ],
                           **fit_kwargs)
    score_func_name: The name of the method used to get the score with signature:
                     -> clf.score(X: Union[np.ndarray, pd.DataFrame],
                                  y: Union[np.ndarray, ],
                                  **forward_kwargs)
    n_splits: Number of split for the kfold.
    model_cls_args: Arguments to give to the model_cls constructor.
    fit_kwargs: kwargs to give to the fit method of model_cls.
    forward_kwargs: kwargs to give to the score method of model_cls.
    save_kwargs_list:
    verbose: True to print some stats else False.
    optimisation_func: The optimisation function with signatue:
                        -> optimisation_func(model_cls, X, y, gen, fit_func_name, score_func_name, n_splits,
                                             model_cls_args, fit_kwargs, forward_kwargs, verbose)
    nb_cpu: number of used cpu. Default: "max".
    Return
    ---------
    The given ParamGens optimized.
    """
    if save_kwargs_list is None:
        save_kwargs_list = [{} for _ in param_gens]
    if forward_kwargs is None:
        forward_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}
    if model_cls_args is None:
        model_cls_args = []

    if len(save_kwargs_list) == 1:
        save_kwargs_list = [save_kwargs_list[0] for _ in param_gens]

    assert len(save_kwargs_list) == len(param_gens)

    if isinstance(nb_cpu, str):
        if nb_cpu.lower() == "max":
            nb_cpu = multiprocessing.cpu_count()
    else:
        nb_cpu = min(multiprocessing.cpu_count(), nb_cpu)

    if verbose:
        logging.info(f"Number of available cpu : {multiprocessing.cpu_count()}, \n"
                     f"Using {nb_cpu} cpu")

    with Pool(nb_cpu) as p:
        outputs = p.starmap(optimisation_func, [
            (model_cls, X, y, gen, fit_func_name, score_func_name, n_splits,
             model_cls_args, fit_kwargs, forward_kwargs, save_kwargs_list[i], verbose)
            for i, gen in enumerate(param_gens)
        ])
    return [param_gen for param_gen in outputs]


class TestingModel:

    hp_possible_values = {
        "hp0": [a for a in "abcdefghijklmnopqrstuvxyz"],
        "hp1": [i ** 2 for i in range(5)],
        "hp2": [str(i**3) for i in range(5)],
        "hp6": ["unique"],
        "hp7": [str(2*i) for i in range(5)],
    }

    hp_bounds = {
        "hp3": [0.0, 5.2, 1],
        "hp4": [-1.3, 42.3, 5],
        "hp5": [2.7, 125.9, 10],
    }

    hp_best_values = {
        "hp0": "x",
        "hp1": 9,
        "hp2": 27,
        "hp3": 3,
        "hp4": 3,
        "hp5": 3,
        "hp6": "unique",
        "hp7": 3,
    }

    def __init__(self, noise=0.0, **hp):
        self.hp = hp
        self.set_default_hp(**TestingModel.hp_best_values)
        self.noise = noise

    def set_hp(self, **hp):
        for hpKey, hpValue in hp.items():
            self.hp[hpKey] = hpValue

    def set_default_hp(self, **hp):
        for hpKey, hpValue in hp.items():
            if hpKey not in self.hp:
                self.hp[hpKey] = hpValue

    def fit(self, *args, **kwargs):
        pass

    def score(self, *args, **kwargs):
        me = 0.0
        for i, (k, v) in enumerate(self.hp.items()):
            # e = (float(v) - float(TestingModel.hp_best_values[k])) / float(TestingModel.hp_best_values[k])
            e = float(v != TestingModel.hp_best_values[k])
            noise = np.random.normal(0, self.noise)
            me = (i * me + noise + e) / (i + 1)
        return 1 - me


if __name__ == '__main__':
    from modules.gp_search import GPOParamGen

    gpo_param_gens = [
        GPOParamGen(
            values_dict=TestingModel.hp_possible_values,
            max_itr=100_000,
            max_seconds=10,
            Lambda=0.5,
            bandwidth=0.5,
        ),
        GPOParamGen(
            values_dict=TestingModel.hp_possible_values,
            max_itr=100_000,
            max_seconds=10,
        )
    ]

    gpo_param_gens = multi_gpo_precessing(
        model_cls=TestingModel,
        X=pd.DataFrame(np.random.random((50, 50))),
        y=np.random.random((50, 1)),
        param_gens=gpo_param_gens,
        n_splits=2,
        model_cls_args=[0.0],
        fit_kwargs={
        },
        forward_kwargs={
        },
        verbose=True,
        nb_cpu="max",
        optimisation_func=optimize_parameters,
    )

    for gpo in gpo_param_gens:
        print('-'*25)
        gpo.save_best_param(save_dir="test_save_opt_hp", save_name="test_opt_hp")
        opt_hp = np.load(f"test_save_opt_hp/test_opt_hp.npy", allow_pickle=True).item()
        print(type(opt_hp), opt_hp)
        print(f"optimal_hp: \n" + '\n'.join([f'{k}: {v}' for k, v in opt_hp.items()]))
        print('-' * 25)

    for gpo in gpo_param_gens:
        gpo.show_expectation()
