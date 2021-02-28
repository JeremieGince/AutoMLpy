import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from typing import Dict, Union, List, Iterable, Tuple
import os
from src.parameter_generators.parameter_generator import ParameterGenerator


class GPOHpSearch(ParameterGenerator):
    def __init__(self,
                 values_dict: Union[Dict[Union[int, str], List[Union[int, float]]],
                                    Dict[Union[int, str], Iterable]],
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
        super(GPOHpSearch, self).__init__(values_dict, **kwargs)

        self.xi = kwargs.get("xi", 0.1)
        self.Lambda = kwargs.get("Lambda", 1.0)
        self.bandwidth = kwargs.get("bandwidth", 1.0)

        self.X, self.y = [], []
        self.gpr = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.Lambda, optimizer=None)

    def reset(self) -> None:
        """
        Reset the current parameter generator.

        Reset the current_itr, the start_time, the X and y container, and reset the gpr.
        """
        super().reset()

        self.X, self.y = [], []
        self.gpr = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.Lambda, optimizer=None)

    @ParameterGenerator.Decorators.increment_counters
    def get_trial_param(self) -> Dict[str, Union[int, float]]:
        """
        Returned a set of trial parameter.

        Increase the current_itr counter.
        """
        if len(self.X) > 0:
            eis = self.expected_improvement()
            idx = np.argmax(eis)
        else:
            idx = np.random.randint(self.xx.shape[0])

        t_sub_space = self.xx[idx]
        t_params = self.convert_subspace_to_param(t_sub_space)
        return t_params

    def get_best_param(self, **kwargs):
        """
        Get the best predicted parameters with the current exploration.
        """
        f_hat = self.gpr.predict(self.xx)
        if kwargs.get("from_history", True):
            f_hat_max = np.max(f_hat)
            history_max = max(self.history, key=lambda t: t[-1])
            if history_max[1] >= f_hat_max:
                return history_max[0]
        b_sub_space = self.xx[np.argmax(f_hat)]
        b_params = self.convert_subspace_to_param(b_sub_space)
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

        self.X.append([param[p] for p in self._values_names])
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
        for i, param_name in enumerate(self._values_names):
            if len(self._values_dict[param_name]) > 1:
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
        assert param_name in self._values_names
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
        dim = self._values_names.index(param_name)
        _x = np.unique(self.xx[:, dim])
        f_hat, std_hat = self.gpr.predict(self.xx, return_std=True)
        f_hat = f_hat.reshape((*_x.shape, -1))
        std_hat = std_hat.reshape((*_x.shape, -1))
        mean_dim_f_hat = np.mean(f_hat, axis=-1)
        mean_dim_std_hat = np.mean(std_hat, axis=-1)

        raw_x_dim, raw_y = np.array(self.X)[:, dim], np.array(self.y)
        return _x, mean_dim_f_hat, mean_dim_std_hat, (raw_x_dim, raw_y)
