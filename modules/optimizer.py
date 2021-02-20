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

    def fit_model_(
            self,
            model: object,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            X_val: Union[np.ndarray, pd.DataFrame, torch.Tensor] = None,
            y_val: Union[np.ndarray, torch.Tensor] = None,
            **hp
    ) -> object:
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
                        self.fit_model_(model, sub_X_train, sub_y_train, **params)

                        score, _ = self.score(model, sub_X_test, sub_y_test, **params)
                        mean_score = (j * mean_score + score) / (j + 1)
                    except Exception as e:
                        logging.error(str(e))

                if verbose:
                    logging.info(f"\ntrial_params: {params} --> mean score: {mean_score:.3f}")
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


if __name__ == '__main__':
    from modules.grid_search import GridHpSearch
    from modules.random_search import RandomHpSearch
    from tests.pytorch_items.pytorch_datasets import get_MNIST_X_y, get_Cifar10_X_y
    import time
    from tests.pytorch_items.poutyne_hp_optimizers import PoutyneCifar10HpOptimizer, PoutyneMNISTHpOptimizer
    import numpy as np

    from modules.logging_tools import logs_file_setup, log_device_setup, DeepLib

    logs_file_setup(__file__)
    log_device_setup(DeepLib.Pytorch)

    cifar10_X_y_dict = get_Cifar10_X_y()
    cifar10_hp_optimizer = PoutyneCifar10HpOptimizer()

    hp_space = dict(
        epochs=list(range(1, 26)),
        batch_size=[32, 64],
        learning_rate=[10 ** e for e in [-3, -2, -1]],
        nesterov=[True, False],
        momentum=np.linspace(0.01, 0.99, 50),
        # use_batchnorm=[True, False],
        pre_normalized=[True, False],
    )
    param_gen = RandomHpSearch(hp_space, max_seconds=60 * 60 * 0.5, max_itr=10)

    start_time = time.time()
    param_gen = cifar10_hp_optimizer.optimize(
        param_gen,
        cifar10_X_y_dict["train"]["x"],
        cifar10_X_y_dict["train"]["y"],
        n_splits=2,
        save_kwargs=dict(save_name=f"cifar10_hp_opt"),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    opt_hp = param_gen.get_best_param()

    logging.info(f"Predicted best hyper-parameters: \n{param_gen.get_best_params_repr()}")
    model = cifar10_hp_optimizer.build_model(**opt_hp)
    cifar10_hp_optimizer.fit_model_(
        model,
        cifar10_X_y_dict["train"]["x"],
        cifar10_X_y_dict["train"]["y"],
        verbose=True,
        **opt_hp
    )

    test_acc, _ = cifar10_hp_optimizer.score(
        model,
        cifar10_X_y_dict["test"]["x"],
        cifar10_X_y_dict["test"]["y"],
        **opt_hp
    )

    param_gen.write_optimization_to_html(show=True, save_name="cifar10", title="Cifar10")

    logging.info(f"test accuracy: {test_acc*100:.3f}%")
