import logging
import os
from typing import Dict, Union, List, Iterable, Tuple, Type
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from .parameter_generator import ParameterGenerator


class MLPEpsilonGreedySearch(ParameterGenerator):
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
        self.mlp (MLPRegressor): The mlp used to make the predictions.
        """
        super(MLPEpsilonGreedySearch, self).__init__(values_dict, **kwargs)

        self.xi = kwargs.get("xi", 0.1)
        self.xi_decay = kwargs.get("xi_decay", 0.0)
        self.nb_exploration_itr = kwargs.get("nb_exploration_itr", 10)
        self._hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (100, 100, ))

        self.X, self.y = [], []
        self.mlp = self._make_default_mlp()

        # --------- html data --------- #
        self._expectation_of_params = {}

    def _make_default_mlp(self):
        return MLPRegressor(
            hidden_layer_sizes=self._hidden_layer_sizes,
            learning_rate="adaptive",
            early_stopping=False,
        )

    def reset(self) -> None:
        """
        Reset the current parameter generator.

        Reset the current_itr, the start_time, the X and y container, and reset the gpr.
        """
        super().reset()

        self.X, self.y = [], []
        self.mlp = self._make_default_mlp()

    @ParameterGenerator.Decorators.increment_counters
    def get_trial_param(self) -> Dict[str, Union[int, float]]:
        """
        Returned a set of trial parameter.

        Increase the current_itr counter.
        """
        xi = self.xi * (1 - self.xi_decay) ** (self.current_itr - self.nb_exploration_itr)
        if self.current_itr < self.nb_exploration_itr:
            xi = np.inf
        if len(self.X) >= 2 and np.random.random() > xi:
            return self.get_best_param(from_history=False)
        else:
            idx = np.random.randint(self.xx.shape[0])

        t_sub_space = self.xx[idx]
        t_params = self.convert_subspace_to_param(t_sub_space)
        return t_params

    def get_best_param(self, **kwargs):
        """
        Get the best predicted parameters with the current exploration.
        """
        if self.minimise:
            return self._get_best_param_minimise(**kwargs)
        else:
            return self._get_best_param_maximise(**kwargs)

    def _get_best_param_maximise(self, **kwargs):
        """
        Get the best (max) predicted parameters with the current exploration.
        """
        f_hat = self.mlp.predict(self.xx)
        if kwargs.get("from_history", True):
            f_hat_max = np.max(f_hat)
            history_max = max(self.history, key=lambda t: t[-1])
            if history_max[1] >= f_hat_max:
                return history_max[0]
        b_sub_space = self.xx[np.argmax(f_hat)]
        b_params = self.convert_subspace_to_param(b_sub_space)
        return b_params

    def _get_best_param_minimise(self, **kwargs):
        """
        Get the best (min) predicted parameters with the current exploration.
        """
        f_hat = self.mlp.predict(self.xx)
        if kwargs.get("from_history", True):
            f_hat_min = np.min(f_hat)
            history_min = min(self.history, key=lambda t: t[-1])
            if history_min[1] <= f_hat_min:
                return history_min[0]
        b_sub_space = self.xx[np.argmin(f_hat)]
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
        self.y.append(score)
        if kwargs.get("mlp_fit", True):
            self.mlp_fit_()

    def mlp_fit_(self):
        try:
            self.mlp.fit(np.array(self.X), np.array(self.y))
        except MemoryError:
            self.mlp = BaggingRegressor(base_estimator=self.mlp, n_estimators=10)
            self.mlp.fit(np.array(self.X), np.array(self.y))
        except ValueError:
            logging.warning("The mlp is not ready to fit. Must be at least 2 training data.")

    def _compute_param_expectation(
            self,
            param_name: str
    ) -> dict:
        """
        Get the expectation of hp-score_space.

        _x (np.ndarray): The score_space of the given parameter.
        f_hat (np.ndarray): Predicted score of the given hp score_space.
        raw_x_dim (np.ndarray): X trial score_space of the given parameter.
        raw_y (np.ndarray): Score of the trial score_space of the given parameter.

        Parameters
        ----------
        param_name: The parameter name to show its expectation.

        Return
        ---------
        dict of keys and values: [_x, f_hat, raw_x_dim, raw_y]
        """
        dim = self._values_names.index(param_name)
        _x = np.unique(self.xx[:, dim])
        f_hat = self.mlp.predict(self.xx)
        raw_x_dim, raw_y = np.array(self.X)[:, dim], np.array(self.y)
        return dict(
            _x=_x,
            raw_x_dim=raw_x_dim,
            raw_y=raw_y,
            x_dim=np.unique(raw_x_dim),
            f_hat=f_hat,
        )

    def _init_html_fig(self, x_y_dict: Dict, **kwargs) -> go.Figure:
        fig = super(MLPEpsilonGreedySearch, self)._init_html_fig(x_y_dict, **kwargs)

        self._expectation_of_params = {}

        for p in self._values_names:
            self._expectation_of_params[p] = self._compute_param_expectation(p)

        # f_hat
        fig.add_trace(
            go.Scatter(x=self._expectation_of_params[self._values_names[0]]['_x'],
                       y=self._expectation_of_params[self._values_names[0]]['f_hat'],
                       mode='lines',
                       name="Expectation",
                       line=dict(width=0.5, color='rgba(255, 0, 0, 1.0)'), ),
        )

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
                                    ],
                                    y=[
                                        x_y_dict[p_name]['y'],
                                        self._expectation_of_params[p_name]['f_hat'],
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
