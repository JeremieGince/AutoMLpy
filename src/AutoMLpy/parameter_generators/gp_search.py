import os
from typing import Dict, Union, List, Iterable, Tuple, Type
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF, RationalQuadratic

from .parameter_generator import ParameterGenerator


class GPOHpSearch(ParameterGenerator):
    def __init__(self,
                 values_dict: Union[Dict[Union[int, str], List[Union[int, float]]],
                                    Dict[Union[int, str], Iterable]],
                 **kwargs):
        """
        Used to generate the hyper-parameter (hp) score_space, generate trial parameter for the exploration and get the best
        set of hp of the hp score_space according to the current exploration.

        Parameters
        ----------
        values_dict:
            A dictionary which contained all the possible values of each hyper-parameter
            used to generate the exploring score_space.
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
        self.xx (np.ndarray): Observation score_space.
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
        self.gpr_n_restarts_optimizer = kwargs.get("gpr_n_restarts_optimizer", 10)
        self._default_kernel = kwargs.get("kernel", self._make_default_kernel(**kwargs))
        self._kernel = deepcopy(self._default_kernel)
        self._kernel_optimizer = kwargs.get("kernel_optimizer", "fmin_l_bfgs_b")

        self.X, self.y = [], []
        self.X_transformed = []
        self._current_xx_transformed_pred = None
        self._current_X_transformed_pred = None
        self._is_fitted = False
        self.estimator = self._make_default_gpr()

        # --------- html data --------- #
        self._expectation_of_params = {}

    def _make_default_kernel(self, **kwargs):
        sub_kernels = []
        C = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-5, 1e3))
        if kwargs.get("kernel_add_matern", True):
            k0 = Matern(length_scale=self.bandwidth, nu=5 / 2)
            sub_kernels.append(deepcopy(C) * k0)
        if kwargs.get("kernel_add_rbf", True):
            k1 = RBF(length_scale=self.bandwidth)
            sub_kernels.append(deepcopy(C) * k1)
        if kwargs.get("kernel_add_rational_quadratic", True):
            k2 = RationalQuadratic(length_scale=self.bandwidth)
            sub_kernels.append(deepcopy(C) * k2)
        kernel = sum(sub_kernels)
        return kernel

    def _make_default_gpr(self):
        return GaussianProcessRegressor(
            self._kernel,
            alpha=self.Lambda,
            optimizer=self._kernel_optimizer,
            n_restarts_optimizer=self.gpr_n_restarts_optimizer,
            copy_X_train=False,
        )

    def reset(self) -> None:
        """
        Reset the current parameter generator.

        Reset the current_itr, the start_time, the X and y container, and reset the gpr.
        """
        super().reset()

        self.X, self.y = [], []
        self._kernel = deepcopy(self._default_kernel)
        self.estimator = self._make_default_gpr()

    @ParameterGenerator.Decorators.increment_counters
    def get_trial_param(self, worker_id: int = -1) -> Dict[str, Union[int, float]]:
        """
        Returned a set of trial parameter.

        Increase the current_itr counter.
        """
        if self._is_fitted:
            eis, idx = self.expected_improvement(self.get_xi(worker_id))
        else:
            idx = np.random.randint(self.xx.transformed_space.shape[0])

        t_sub_space = self.xx.inverse_transform(self.xx.transformed_space[idx])
        t_params = self.convert_subspace_to_param(t_sub_space)
        return t_params

    def get_xi(self, worker_id: int = -1):
        if worker_id == -1:
            return self.xi
        return np.linspace(1e-2, 1.0, num=self._nb_workers)[worker_id]

    def get_best_param(self, **kwargs):
        """
        Get the best predicted parameters with the current exploration.
        """
        if not self._is_fitted:
            self.estimator_fit_()
        if self.minimise:
            return self._get_best_param_minimise(**kwargs)
        else:
            return self._get_best_param_maximise(**kwargs)

    def _get_best_param_maximise(self, **kwargs):
        """
        Get the best (max) predicted parameters with the current exploration.
        """
        f_hat, _ = self._current_xx_transformed_pred
        if kwargs.get("from_history", True):
            history_max = max(self.history, key=lambda t: t[-1])
            return history_max[0]
        b_sub_space = self.xx.inverse_transform(self.xx.transformed_space[np.argmin(f_hat)])
        b_params = self.convert_subspace_to_param(b_sub_space)
        return b_params

    def _get_best_param_minimise(self, **kwargs):
        """
        Get the best (min) predicted parameters with the current exploration.
        """
        f_hat, _ = self._current_xx_transformed_pred
        if kwargs.get("from_history", True):
            history_min = min(self.history, key=lambda t: t[-1])
            return history_min[0]
        b_sub_space = self.xx.inverse_transform(self.xx.transformed_space[np.argmin(f_hat)])
        b_params = self.convert_subspace_to_param(b_sub_space)
        return b_params

    def add_score_info(self, param, score, **kwargs):
        """
        Add the result of the trial parameters.

        Parameters
        ----------
        param: The trial parameters.
        score: The associated score of the trial parameters.
        """
        super().add_score_info(param, score)

        self.X.append([param[p] for p in self._values_names])
        self.X_transformed.append(self.xx.transform(self.X[-1]))
        self.y.append(score)
        if kwargs.get("fit", True):
            self.estimator_fit_()

    def estimator_fit_(self, worker_id=-1):
        self.estimator.fit(np.array(self.X_transformed), np.array(self.y))
        self._current_xx_transformed_pred = self.estimator.predict(
            self.xx.transformed_space.to_numpy(), return_std=True
        )
        self._current_X_transformed_pred = self.estimator.predict(
            np.array(self.X_transformed), return_std=True
        )
        self._is_fitted = True

    def expected_improvement(self, xi: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.minimise:
            eis = self.expected_improvement_minimise(xi)
            idx = np.argmax(eis)
        else:
            eis = self.expected_improvement_maximise(xi)
            idx = np.argmin(eis)
        return eis, idx

    def expected_improvement_maximise(self, xi: float) -> np.ndarray:
        """
        Returned the expected improvement of the search score_space.
        """
        f_hat, _ = self._current_X_transformed_pred
        best_f = np.max(f_hat)

        f_hat, std_hat = self._current_xx_transformed_pred
        improvement = f_hat - best_f - xi

        Z = improvement / std_hat
        ei = improvement * norm.cdf(Z) + std_hat * norm.pdf(Z)
        return ei

    def expected_improvement_minimise(self, xi: float) -> np.ndarray:
        """
        Returned the expected improvement of the search score_space.
        """
        f_hat, _ = self._current_X_transformed_pred
        f_best = np.min(f_hat)

        mu_hat, std_hat = self._current_xx_transformed_pred
        gamma = (f_best - mu_hat) / std_hat

        ei = std_hat * (gamma * norm.cdf(gamma) + np.random.normal(gamma, xi))
        return ei

    def show_expectation(self, **kwargs):
        """
        Show the expectation of hp-score_space.

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
        warnings.warn("Deprecated --> must not work as it should be. Will be removed in the future", DeprecationWarning)
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
            _x, mean_dim_f_hat, mean_dim_std_hat, (raw_x_dim, raw_y) = self._compute_expectation_of_param(param_name)
            x_dim = np.unique(raw_x_dim)
            subfig.plot(_x, mean_dim_f_hat)
            subfig.plot(raw_x_dim, raw_y, 'x')
            subfig.fill_between(_x, mean_dim_f_hat, mean_dim_f_hat + mean_dim_std_hat, alpha=0.4)
            # subfig.fill_between(_x, mean_dim_f_hat, mean_dim_f_hat - mean_dim_std_hat, alpha=0.4)

            if param_name in self._param_idx_to_name:
                subfig.set_xticks(list(range(len(x_dim))))
                subfig.set_xticklabels(self.convert_idx_to_param(param_name, x_dim))
            subfig.set_xlabel("hp score_space [-]")
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
        Show the expectation of hp-score_space.

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
        warnings.warn("Deprecated --> must not work as it should be. Will be removed in the future", DeprecationWarning)
        assert param_name in self._values_names
        _x, mean_dim_f_hat, mean_dim_std_hat, (raw_x_dim, raw_y) = self._compute_expectation_of_param(param_name)
        x_dim = np.unique(raw_x_dim)

        plt.figure(1)
        plt.plot(_x, mean_dim_f_hat)
        plt.plot(raw_x_dim, raw_y, 'x')
        plt.fill_between(_x, mean_dim_f_hat, mean_dim_f_hat + mean_dim_std_hat, alpha=0.4)
        plt.fill_between(_x, mean_dim_f_hat, mean_dim_f_hat - mean_dim_std_hat, alpha=0.4)

        if param_name in self._param_idx_to_name:
            plt.xticks(list(range(len(x_dim))), self.convert_idx_to_param(param_name, x_dim))
        plt.xlabel("hp score_space [-]")
        plt.ylabel("Expected score [-]")
        plt.title(f"EI of {param_name}")

        fig_dir = kwargs.get('fig_dir', 'Figures/')
        os.makedirs(f"{fig_dir}", exist_ok=True)
        plt.savefig(f"{fig_dir}/{kwargs.get('save_name', 'expectation.png')}", dpi=300)
        plt.show()

    def _compute_expectation_of_param(self, param_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                      Tuple[np.ndarray, np.ndarray]]:
        """
        Get the expectation of hp-score_space.

        _x (np.ndarray): The score_space of the given parameter.
        mean_dim_f_hat (np.ndarray): Predicted score of the given hp score_space.
        mean_dim_std_hat (np.ndarray): Predicted score std of the given hp score_space.
        raw_x_dim (np.ndarray): X trial score_space of the given parameter.
        raw_y (np.ndarray): Score of the trial score_space of the given parameter.

        Parameters
        ----------
        param_name: The parameter name to show its expectation.

        Return
        ---------
        _x, mean_dim_f_hat, mean_dim_std_hat, (raw_x_dim, raw_y)
        """
        warnings.warn("Deprecated --> must not work as it should be. Will be removed in the future", DeprecationWarning)
        dim = self._values_names.index(param_name)
        _x = np.unique(self.xx[:, dim])
        f_hat, std_hat = self._current_xx_transformed_pred
        f_hat = f_hat.reshape((*_x.shape, -1))
        std_hat = std_hat.reshape((*_x.shape, -1))
        mu_hat = np.mean(f_hat, axis=-1)
        std_hat = np.mean(std_hat, axis=-1)

        raw_x_dim, raw_y = np.array(self.X)[:, dim], np.array(self.y)
        return _x, mu_hat, std_hat, (raw_x_dim, raw_y)

    def _compute_param_expectation(
            self,
            param_name: str
    ) -> dict:
        """
        Get the expectation of hp-score_space.

        _x (np.ndarray): The score_space of the given parameter.
        mu_hat (np.ndarray): Predicted score of the given hp score_space.
        std_hat (np.ndarray): Predicted score std of the given hp score_space.
        raw_x_dim (np.ndarray): X trial score_space of the given parameter.
        raw_y (np.ndarray): Score of the trial score_space of the given parameter.

        Parameters
        ----------
        param_name: The parameter name to show its expectation.

        Return
        ---------
        dict of keys and values: [_x, mu_hat, std_hat, raw_x_dim, raw_y]
        """
        dim = self._values_names.index(param_name)
        _x = np.unique(self.xx.get_random_subspace(1_000).to_numpy()[:, dim])
        f_hat, _ = self._current_X_transformed_pred
        mu_hat, std_hat = self._current_xx_transformed_pred
        raw_x_dim, raw_y = np.array(self.X)[:, dim], np.array(self.y)
        ei, _ = self.expected_improvement(self.get_xi())
        return dict(
            _x=_x,
            mu_hat=mu_hat,
            std_hat=std_hat,
            raw_x_dim=raw_x_dim,
            raw_y=raw_y,
            x_dim=np.unique(raw_x_dim),
            ei=ei,
            f_hat=f_hat,
        )

    def _init_html_fig(self, x_y_dict: Dict, **kwargs) -> go.Figure:
        fig = super(GPOHpSearch, self)._init_html_fig(x_y_dict, **kwargs)

        self._expectation_of_params = {}

        for p in self._values_names:
            self._expectation_of_params[p] = self._compute_param_expectation(p)

        # Mean
        fig.add_trace(
            go.Scatter(x=self._expectation_of_params[self._values_names[0]]['_x'],
                       y=self._expectation_of_params[self._values_names[0]]['mu_hat'],
                       mode='lines',
                       name="Mean expectation",
                       line=dict(width=0.5, color='rgba(255, 0, 0, 1.0)'), ),
        )
        # Std
        fig.add_trace(
            go.Scatter(
                x=list(self._expectation_of_params[self._values_names[0]]['_x'])
                  + list(self._expectation_of_params[self._values_names[0]]['_x'])[::-1],
                y=list(self._expectation_of_params[self._values_names[0]]['mu_hat']
                       - self._expectation_of_params[self._values_names[0]]['std_hat'])
                  + list(self._expectation_of_params[self._values_names[0]]['mu_hat']
                         + self._expectation_of_params[self._values_names[0]]['std_hat'])[::-1],
                mode='lines',
                fill="toself",
                fillcolor='rgba(255, 0, 0, 0.05)',
                name="Std expectation",
                line=dict(width=0.0),
            )
        )
        # EI
        fig.add_trace(
            go.Scatter(x=self._expectation_of_params[self._values_names[0]]['_x'],
                       y=self._expectation_of_params[self._values_names[0]]['ei'],
                       mode='lines',
                       name="Expected improvement",
                       line=dict(width=0.5, color='rgba(0, 0, 255, 0.8)'), ),
        )
        # f_hat
        # fig.add_trace(
        #     go.Scatter(x=self._expectation_of_params[self._values_names[0]]['raw_x_dim'],
        #                y=self._expectation_of_params[self._values_names[0]]['f_hat'],
        #                mode='markers',
        #                name="f_hat",
        #                line=dict(width=0.5, color='rgba(0, 255, 0, 0.8)'), ),
        # )

        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))

        return fig

    def _add_dropdown_html_fig_(self, fig, x_y_dict, **kwargs):
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[
                                dict(
                                    x=[
                                        x_y_dict[p_name]['x'],
                                        self._expectation_of_params[p_name]['_x'],
                                        list(self._expectation_of_params[p_name]['_x'])
                                        + list(self._expectation_of_params[p_name]['_x'])[::-1],
                                        self._expectation_of_params[p_name]['_x']
                                    ],
                                    y=[
                                        x_y_dict[p_name]['y'],
                                        self._expectation_of_params[p_name]['mu_hat'],
                                        list(self._expectation_of_params[p_name]['mu_hat']
                                             - self._expectation_of_params[p_name]['std_hat'])
                                        + list(self._expectation_of_params[p_name]['mu_hat']
                                               + self._expectation_of_params[p_name]['std_hat'])[::-1],
                                        self._expectation_of_params[p_name]['ei']
                                    ],
                                    marker=dict(color=x_y_dict[p_name]['y'], showscale=True),
                                ),
                                {
                                    # "title": f"{p_name}",
                                    "xaxis.title.text": f"{p_name}: parameter space [-]",
                                    "yaxis.title.text": "Score [-]",
                                }
                            ],
                            label=p_name,
                            method="update"
                        )
                        for p_name in self._values_names
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.9,
                    xanchor="left",
                    y=1.1,
                    yanchor="middle"
                ),
            ]
        )
