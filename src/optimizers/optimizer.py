import warnings
import multiprocessing
import numpy as np
import tqdm
from sklearn.model_selection import KFold
from typing import Union, List, Tuple
import pandas as pd
from multiprocessing import Pool
import torch
from copy import deepcopy
import logging
from src.parameter_generators.parameter_generator import ParameterGenerator


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
            nb_workers: Union[int, str] = "max",
            save_kwargs: dict = None,
            verbose: bool = True,
            **kwargs
    ):
        """
        Optimize hyper-paremeters using the given ParameterGenerator and kfolding.

        Parameters
        ----------
        param_gen: The parameter generator.
        X: The training data.
        y: The training labels.
        n_splits: Number of split for the kfold.
        nb_workers: number of used cpu. Default: "max".
        save_kwargs:
        verbose: True to print some stats else False.

        Return
        ---------
        The given ParameterGenerator optimized.
        """
        if save_kwargs is None:
            save_kwargs = {}
        warnings.simplefilter("ignore", UserWarning)

        nb_workers = self._setup_nb_workers(nb_workers, verbose, **kwargs)

        stop_criterion = kwargs.get("stop_criterion", None)
        stop_criterion_trigger = False

        param_gen.reset()
        max_itr = len(param_gen)

        if verbose:
            progress = tqdm.tqdm(range(max_itr), unit='itr', postfix="optimisation")
        else:
            progress = range(max_itr)
        for i in progress:
            if not bool(param_gen) or stop_criterion_trigger:
                break
            try:
                if nb_workers > 1:
                    workers_required = min(nb_workers, max_itr - i)
                    outputs = self._execute_multiple_iteration_on_param_gen(workers_required, param_gen, X, y, n_splits)
                else:
                    outputs = [self._execute_iteration_on_param_gen(param_gen, X, y, n_splits)]

                for (params, mean_score) in outputs:
                    param_gen.add_score_info(params, mean_score)

                    if verbose:
                        logging.debug(f"\ntrial_params: {params} --> mean score: {mean_score:.3f}")
                        progress.set_postfix_str(f"mean_score: {mean_score:.2f}")
                        progress.update()

                    if stop_criterion is not None and stop_criterion <= mean_score:
                        if verbose:
                            logging.info(f"Early stopping ->"
                                         f" itr: {param_gen.current_itr},"
                                         f" elapse_time: {param_gen.elapse_time:.3f} [s],"
                                         f" mean score: {mean_score:.3f}")
                        stop_criterion_trigger = True
            except Exception as ee:
                logging.error(str(ee))
                raise ee

        if verbose:
            progress.close()
        param_gen.save_best_param(**save_kwargs)
        param_gen.write_optimization_to_html(show=False, **save_kwargs)
        return param_gen

    @staticmethod
    def _setup_nb_workers(
            nb_workers: Union[int, str] = "max",
            verbose: bool = True,
            **kwargs
    ):
        if isinstance(nb_workers, str):
            if nb_workers.lower() == "max":
                nb_workers = multiprocessing.cpu_count()
        else:
            nb_workers = min(multiprocessing.cpu_count(), nb_workers)

        if verbose:
            logging.info(f"Number of available cpu : {multiprocessing.cpu_count()} --> Using {nb_workers} cpu")
        return nb_workers

    def _execute_multiple_iteration_on_param_gen(
            self,
            nb_workers: int,
            param_gen: ParameterGenerator,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            n_splits: int = 2,
    ):
        params_list = [param_gen.get_trial_param() for _ in range(nb_workers)]
        with Pool(nb_workers) as p:
            scores = p.starmap(self._try_params, [
                (params, X, y, n_splits)
                for params in params_list
            ])
        outputs = [(params, score) for params, score in zip(params_list, scores)]
        return outputs

    def _execute_iteration_on_param_gen(
            self,
            param_gen: ParameterGenerator,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            n_splits: int = 2,
    ):
        params = param_gen.get_trial_param()
        mean_score = self._try_params(params, X, y, n_splits)
        return params, mean_score

    def _try_params(
            self,
            params,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            n_splits: int = 2,
    ) -> float:
        kf = KFold(n_splits=n_splits, shuffle=True)

        mean_score = 0.0
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
                raise e

        return mean_score


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
                -> __init__(*model_cls_args, **x)
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
                -> __init__(*model_cls_args, **x)
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
    from src.parameter_generators.random_search import RandomHpSearch
    from tests.pytorch_items.pytorch_datasets import get_Cifar10_X_y
    import time
    from tests.pytorch_items.poutyne_hp_optimizers import PoutyneCifar10HpOptimizer
    import numpy as np

    from src.logging_tools import logs_file_setup, log_device_setup, DeepLib

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
    _param_gen = RandomHpSearch(hp_space, max_seconds=60 * 60 * 0.5, max_itr=10)

    start_time = time.time()
    _param_gen = cifar10_hp_optimizer.optimize(
        _param_gen,
        cifar10_X_y_dict["train"]["x"],
        cifar10_X_y_dict["train"]["y"],
        n_splits=2,
        save_kwargs=dict(save_name=f"cifar10_hp_opt"),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    opt_hp = _param_gen.get_best_param()

    logging.info(f"Predicted best hyper-parameters: \n{_param_gen.get_best_params_repr()}")
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

    _param_gen.write_optimization_to_html(show=True, save_name="cifar10", title="Cifar10")

    logging.info(f"test accuracy: {test_acc*100:.3f}%")
