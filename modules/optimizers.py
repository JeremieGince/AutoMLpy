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
import torch
from multiprocessing import Pool
from modules.models.torch_model import TorchModel
import time
import os
import logging


class ParamGen:
    def __init__(self,
                 bounds_dict: Union[Dict[Union[int, str], List[Union[int, float]]],
                                    Dict[Union[int, str], Iterable]],
                 default_interval: int = 1,
                 bounds_as_list_of_possible_values: bool = False,
                 **kwargs):
        """
        Used to generate the hyper-parameter (hp) space, generate trial parameter for the exploration and get the best
        set of hp of the hp space according to the current exploration.

        Parameters
        ----------
        bounds_dict:
            A bounds dictionary used to generate the exploring space.
        default_interval:
            The default interval to sample the hp space.
        bounds_as_list_of_possible_values:
            True if the bounds dict values must be used as a lists of possible values for the hyper-parameter.
        kwargs: {

        }

        Attributes
        ----------
        self.history List[Tuple]: The history of the hp search.
        """
        self._bounds_names = list(bounds_dict.keys())
        self._bounds_dict = bounds_dict
        self._bounds_as_list_of_possible_values = bounds_as_list_of_possible_values
        self._default_interval = default_interval
        self.history = []

    @property
    def bounds_names(self) -> List[str]:
        return self._bounds_names

    def reset(self) -> None:
        """
        Reset the current parameter generator.
        """
        pass

    def __len__(self) -> int:
        """
        Returned the number of trial that the current param gen can make.
        """
        raise NotImplementedError()

    def __bool__(self) -> bool:
        """
        Returned True if the current param gen can return trial parameters.
        """
        return True

    def get_trial_param(self) -> Dict[str, Union[int, float]]:
        """
        Returned a set of trial parameter.
        """
        raise NotImplementedError()

    def get_best_param(self) -> Dict[str, Union[int, float]]:
        """
        Get the best predicted parameters with the current exploration.
        """
        return max(self.history, key=lambda t: t[-1])

    def add_score_info(self, param: Dict[str, Union[int, float]], score: float) -> None:
        """
        Add the result of the trial parameters.

        Parameters
        ----------
        param: The trial parameters.
        score: The associated score of the trial parameters.
        """
        self.history.append((param, score))

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


class GridParamGen(ParamGen):
    def __init__(self, bounds_dict, default_interval=1, bounds_as_list_of_possible_values=False):
        super(GridParamGen, self).__init__(bounds_dict, default_interval, bounds_as_list_of_possible_values)
        if bounds_as_list_of_possible_values:
            xx = np.meshgrid(*[bounds_dict[p] for p in self._bounds_names])
        else:
            xx = np.meshgrid(*[
                np.arange(
                    bounds_dict[p][0],
                    bounds_dict[p][1] + (bounds_dict[p][2] if len(bounds_dict[p]) > 2 else default_interval),
                    bounds_dict[p][2] if len(bounds_dict[p]) > 2 else default_interval
                ) for p in self._bounds_names
            ])
        self.params = list(zip(*[_xx.ravel() for _xx in xx]))
        self.idx = 0

    def reset(self):
        self.idx = 0

    def __len__(self):
        return len(self.params)

    def get_trial_param(self):
        param = self.params[self.idx]
        self.idx += 1
        return {self._bounds_names[i]: param[i] for i in range(len(param))}


class RandomParamGen(ParamGen):
    def __init__(self, bounds_dict, default_interval=1, bounds_as_list_of_possible_values=False):
        super(RandomParamGen, self).__init__(bounds_dict, default_interval, bounds_as_list_of_possible_values)

    def __len__(self):
        pass

    def get_trial_param(self):
        pass


class GPOParamGen(ParamGen):
    def __init__(self,
                 bounds_dict: Union[Dict[Union[int, str], List[Union[int, float]]],
                                    Dict[Union[int, str], Iterable]],
                 default_interval: int = 1,
                 bounds_as_list_of_possible_values: bool = False,
                 **kwargs):
        """
        Used to generate the hyper-parameter (hp) space, generate trial parameter for the exploration and get the best
        set of hp of the hp space according to the current exploration.

        Parameters
        ----------
        bounds_dict:
            A bounds dictionary used to generate the exploring space.
        default_interval:
            The default interval to sample the hp space.
        bounds_as_list_of_possible_values:
            True if the bounds dict values must be used as a lists of possible values for the hyper-parameter.
        kwargs: {
                    xi (float): Exploration parameter. Must be in [0, 1]. default: 0.1.
                    Lambda (float): Default: 1.0.
                    bandwidth (float): lenght_scale of the RBF kernel used in self.gpr. Default: 1.0.
                    max_itr (int): Max iteration of the gpo. Default: 30.
                    max_seconds (int): Max seconds that the gpo can take to make its optimisation. Default: 60**2.
        }

        Attributes
        ----------
        self.history List[Tuple]: The history of the hp search.
        self._param_name_to_idx (Dict[str, int]) : Dict container for string bounds element used to convert str to idx.
        self._param_idx_to_name (Dict[int, str]) : Dict container for string bounds element used to convert back idx
                                                   to str.
        self.xx (np.ndarray): Observation space.
        self.current_itr (int): Current iteration of the gpo.
        self.start_time (int): Starting time of the optimisation.
        self.X (List): List of the trial hp.
        self.y (List): List of associated score of the trial hp (self.X).
        self.gpr (GaussianProcessRegressor): The gpr used to make the predictions.
        """
        super(GPOParamGen, self).__init__(bounds_dict, default_interval, bounds_as_list_of_possible_values)
        self._param_name_to_idx = {}
        self._param_idx_to_name = {}
        if bounds_as_list_of_possible_values:
            for p in self._bounds_names:
                if self.check_str_in_iterable(bounds_dict[p]):
                    self.add_conversion_tables_param_name_to_idx(p, bounds_dict[p])
                    bounds_dict[p] = self.convert_param_to_idx(p, bounds_dict[p])
            self.xx = np.meshgrid(*[bounds_dict[p] for p in self._bounds_names])
        else:
            self.xx = np.meshgrid(*[
                np.arange(
                    bounds_dict[p][0],
                    bounds_dict[p][1] + (bounds_dict[p][2] if len(bounds_dict[p]) > 2 else default_interval),
                    bounds_dict[p][2] if len(bounds_dict[p]) > 2 else default_interval
                ) for p in self._bounds_names
            ])
        self.xx = np.array(list(zip(*[_x.ravel() for _x in self.xx])))

        self.xi = kwargs.get("xi", 0.1)
        self.Lambda = kwargs.get("Lambda", 1.0)
        self.bandwidth = kwargs.get("bandwidth", 1.0)

        self.max_itr = kwargs.get("max_itr", 30)
        self.current_itr = 0
        self.max_seconds = kwargs.get("max_seconds", 60**2)
        self.start_time = time.time()
        self.X, self.y = [], []
        self.gpr = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.Lambda, optimizer=None)

    def reset(self) -> None:
        """
        Reset the current parameter generator.

        Reset the current_itr, the start_time, the X and y container, and reset the gpr.
        """
        super().reset()
        self.start_time = time.time()
        self.current_itr = 0
        self.X, self.y = [], []
        self.gpr = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.Lambda, optimizer=None)

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

        Increase the current_itr counter.
        """
        self.current_itr += 1

        if len(self.X) > 0:
            eis = self.expected_improvement()
            idx = np.argmax(eis)
        else:
            idx = np.random.randint(self.xx.shape[0])

        t_param = self.xx[idx]
        t_param = {self._bounds_names[i]: t_param[i] for i in range(len(t_param))}
        for p_name in t_param:
            if p_name in self._param_idx_to_name:
                t_param[p_name] = self._param_idx_to_name[p_name][t_param[p_name]]
        return t_param

    def get_best_param(self):
        """
        Get the best predicted parameters with the current exploration.
        """
        f_hat = self.gpr.predict(self.xx)
        b_param = self.xx[np.argmax(f_hat)]
        b_params = {self._bounds_names[i]: b_param[i] for i in range(len(b_param))}
        for p_name in b_params:
            if p_name in self._param_idx_to_name:
                b_params[p_name] = self._param_idx_to_name[p_name][b_params[p_name]]
        return b_params

    def add_score_info(self, param, score):
        """
        Add the result of the trial parameters.

        Parameters
        ----------
        param: The trial parameters.
        score: The associated score of the trial parameters.
        """
        super().add_score_info(param, score)

        for p_name in param:
            if p_name in self._param_name_to_idx:
                param[p_name] = self._param_name_to_idx[p_name][param[p_name]]

        self.X.append([param[p] for p in self._bounds_names])
        self.y.append(score)
        self.gpr.fit(np.array(self.X), np.array(self.y))

    def expected_improvement(self) -> np.ndarray:
        """
        Returned the expected improvement of the search space.
        """
        f_hat = self.gpr.predict(np.array(self.X))
        best_f = np.max(f_hat)

        f_hat, std_hat = self.gpr.predict(self.xx, return_std=True)
        improvement = f_hat - best_f - self.xi

        Z = improvement / std_hat
        ei = improvement * norm.cdf(Z) + std_hat * norm.pdf(Z)
        return ei

    def show_expectation(self, **kwargs):
        """
        Show the expectation of hp-space.

        Parameters
        ----------
        kwargs: {
                    title (str): The title of the plot. default: ''.
                    fig_dir (str): The directory where to save the output figure. Default: 'Figures/'.
                    save_name (str): The save name of the current figure. Default: 'expectation'.
                                    Must not have the keyword 'jpg', 'png' or others cause the current
                                    figure will be saved as png.
        }
        """
        bounds_to_plot = []
        for i, param_name in enumerate(self._bounds_names):
            if len(self._bounds_dict[param_name]) > 1:
                bounds_to_plot.append(param_name)

        k = int(np.ceil(np.sqrt(len(bounds_to_plot))))
        j = 0
        fig, subfigs = plt.subplots(k, k, tight_layout=True)
        subfigs_list = subfigs.reshape(-1)
        fig.suptitle(f"{kwargs.get('title', '')}", fontsize=16)
        for i, param_name in enumerate(bounds_to_plot):
            subfig = subfigs_list[i]
            _x, mean_dim_f_hat, mean_dim_std_hat, (raw_x_dim, raw_y) = self.get_expectation_of_param(param_name)
            x_dim = np.unique(raw_x_dim)
            subfig.plot(_x, mean_dim_f_hat)
            subfig.plot(raw_x_dim, raw_y, 'x')
            subfig.fill_between(_x, mean_dim_f_hat, mean_dim_f_hat + mean_dim_std_hat, alpha=0.4)
            # subfig.fill_between(_x, mean_dim_f_hat, mean_dim_f_hat - mean_dim_std_hat, alpha=0.4)

            if param_name in self._param_idx_to_name:
                subfig.set_xticks(list(range(len(x_dim))))
                subfig.set_xticklabels(self.convert_idx_to_param(param_name, x_dim))
            subfig.set_xlabel("hp space [-]")
            subfig.set_ylabel("Expected score [-]")
            subfig.set_title(f"EI of {param_name}")
            j = i

        for i in range(j+1, len(subfigs_list)):
            subfig = subfigs_list[i]
            subfig.set_axis_off()

        fig_dir = kwargs.get('fig_dir', 'Figures/')
        os.makedirs(f"{fig_dir}", exist_ok=True)
        plt.savefig(f"{fig_dir}/{kwargs.get('save_name', 'expectation')}.png", dpi=300)
        if kwargs.get("show", True):
            plt.show()

    def show_expectation_of_param(self, param_name: str, **kwargs):
        """
        Show the expectation of hp-space.

        Parameters
        ----------
        param_name: The parameter name to show its expectation.
        kwargs: {
                    fig_dir (str): The directory where to save the output figure. Default: 'Figures/'.
                    save_name (str): The save name of the current figure. Default: 'expectation'.
                                    Must not have the keyword 'jpg', 'png' or others cause the current
                                    figure will be saved as png.
        }
        """
        assert param_name in self._bounds_names
        _x, mean_dim_f_hat, mean_dim_std_hat, (raw_x_dim, raw_y) = self.get_expectation_of_param(param_name)
        x_dim = np.unique(raw_x_dim)

        plt.figure(1)
        plt.plot(_x, mean_dim_f_hat)
        plt.plot(raw_x_dim, raw_y, 'x')
        plt.fill_between(_x, mean_dim_f_hat, mean_dim_f_hat + mean_dim_std_hat, alpha=0.4)
        plt.fill_between(_x, mean_dim_f_hat, mean_dim_f_hat - mean_dim_std_hat, alpha=0.4)

        if param_name in self._param_idx_to_name:
            plt.xticks(list(range(len(x_dim))), self.convert_idx_to_param(param_name, x_dim))
        plt.xlabel("hp space [-]")
        plt.ylabel("Expected score [-]")
        plt.title(f"EI of {param_name}")

        fig_dir = kwargs.get('fig_dir', 'Figures/')
        os.makedirs(f"{fig_dir}", exist_ok=True)
        plt.savefig(f"{fig_dir}/{kwargs.get('save_name', 'expectation.png')}", dpi=300)
        plt.show()

    def get_expectation_of_param(self, param_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                 Tuple[np.ndarray, np.ndarray]]:
        """
        Get the expectation of hp-space.

        _x (np.ndarray): The space of the given parameter.
        mean_dim_f_hat (np.ndarray): Predicted score of the given hp space.
        mean_dim_std_hat (np.ndarray): Predicted score std of the given hp space.
        raw_x_dim (np.ndarray): X trial space of the given parameter.
        raw_y (np.ndarray): Score of the trial space of the given parameter.

        Parameters
        ----------
        param_name: The parameter name to show its expectation.

        Return
        ---------
        _x, mean_dim_f_hat, mean_dim_std_hat, (raw_x_dim, raw_y)
        """
        dim = self._bounds_names.index(param_name)
        _x = np.unique(self.xx[:, dim])
        f_hat, std_hat = self.gpr.predict(self.xx, return_std=True)
        f_hat = f_hat.reshape((*_x.shape, -1))
        std_hat = std_hat.reshape((*_x.shape, -1))
        mean_dim_f_hat = np.mean(f_hat, axis=-1)
        mean_dim_std_hat = np.mean(std_hat, axis=-1)

        raw_x_dim, raw_y = np.array(self.X)[:, dim], np.array(self.y)
        return _x, mean_dim_f_hat, mean_dim_std_hat, (raw_x_dim, raw_y)

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


def optimize_parameters(
        model_cls,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        y: Union[np.ndarray, torch.Tensor],
        param_gen: ParamGen,
        fit_func_name: str = "fit",
        score_func_name: str = "score",
        n_splits: int = 2,
        model_cls_args: List = None,
        fit_kwargs: dict = None,
        forward_kwargs: dict = None,
        save_kwargs: dict = None,
        verbose: bool = True,
) -> ParamGen:
    """
    Optimize hyper-paremeters using the given ParamGen and kfolding.

    Parameters
    ----------
    model_cls: A class model. Must have the following implemented methods:
                -> __init__(*model_cls_args, **params)
    X: The training data.
    y: The training labels.
    param_gen: The parameter generator.
    fit_func_name: The name of the method function to fit the model with the signature:
                    -> fit(X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                           y: Union[np.ndarray, torch.Tensor],
                           **fit_kwargs)
    score_func_name: The name of the method used to get the score with signature:
                     -> clf.score(X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                                  y: Union[np.ndarray, torch.Tensor],
                                  **forward_kwargs) -> Tuple[float, float] i.e: (mean, std)
    n_splits: Number of split for the kfold.
    model_cls_args: Arguments to give to the model_cls constructor.
    fit_kwargs: kwargs to give to the fit method of model_cls.
    forward_kwargs: kwargs to give to the score method of model_cls.
    save_kwargs:
    verbose: True to print some stats else False.

    Return
    ---------
    The given ParamGen optimized.
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
        if not param_gen:
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
                        raise ValueError(f"X must be Union[np.ndarray, torch.Tensor, pd.DataFrame]")

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


def optimize_parameters_from_datasets(
        model_cls,
        X,
        y,
        param_gen: ParamGen,
        fit_func_name: str = "fit",
        score_func_name: str = "score",
        n_splits: int = 2,
        model_cls_args: List = None,
        fit_kwargs: dict = None,
        forward_kwargs: dict = None,
        save_kwargs: dict = None,
        verbose: bool = True,
) -> ParamGen:
    """
    Optimize hyper-paremeters using the given ParamGen and kfolding.

    Parameters
    ----------
    model_cls: A class model. Must have the following implemented methods:
                -> __init__(*model_cls_args, **params)
    X: The training data.
    y: The training labels.
    param_gen: The parameter generator.
    fit_func_name: The name of the method function to fit the model with the signature:
                    -> fit(X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                           y: Union[np.ndarray, torch.Tensor],
                           **fit_kwargs)
    score_func_name: The name of the method used to get the score with signature:
                     -> clf.score(X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                                  y: Union[np.ndarray, torch.Tensor],
                                  **forward_kwargs) -> Tuple[float, float] i.e: (mean, std)
    n_splits: Number of split for the kfold.
    model_cls_args: Arguments to give to the model_cls constructor.
    fit_kwargs: kwargs to give to the fit method of model_cls.
    forward_kwargs: kwargs to give to the score method of model_cls.
    save_kwargs:
    verbose: True to print some stats else False.

    Return
    ---------
    The given ParamGen optimized.
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
        if not param_gen:
            break
        try:
            params = param_gen.get_trial_param()

            mean_score = 0
            for j, ((sub_x_tr, sub_x_val), (sub_y_tr, sub_y_val)) in enumerate(
                    zip(X.kfold(n_splits), y.kfold(n_splits))):
                try:
                    clf = model_cls(*model_cls_args, **params)
                    getattr(clf, fit_func_name)(sub_x_tr, sub_y_tr, **fit_kwargs)

                    score, _ = getattr(clf, score_func_name)(sub_x_val, sub_y_val, **forward_kwargs)
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
    X: Union[np.ndarray, torch.Tensor, pd.DataFrame, object],
    y: Union[np.ndarray, torch.Tensor, object],
    param_gens: List[ParamGen],
    fit_func_name: str = "fit",
    score_func_name: str = "score",
    n_splits: int = 2,
    model_cls_args: List = None,
    fit_kwargs: dict = None,
    forward_kwargs: dict = None,
    save_kwargs_list: List = None,
    verbose: bool = True,
    optimisation_func=optimize_parameters_from_datasets,
    nb_cpu: Union[int, str] = "max",
) -> List[ParamGen]:
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
                    -> fit(X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                           y: Union[np.ndarray, torch.Tensor],
                           **fit_kwargs)
    score_func_name: The name of the method used to get the score with signature:
                     -> clf.score(X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                                  y: Union[np.ndarray, torch.Tensor],
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
        print(f"Number of available cpu : {multiprocessing.cpu_count()}, \n"
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

    gpo_param_gens = [
        GPOParamGen(
            bounds_dict=TestingModel.hp_possible_values,
            default_interval=1,
            bounds_as_list_of_possible_values=True,
            max_itr=100_000,
            max_seconds=10,
            Lambda=0.5,
            bandwidth=0.5,
        ),
        GPOParamGen(
            bounds_dict=TestingModel.hp_bounds,
            default_interval=1,
            bounds_as_list_of_possible_values=False,
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
