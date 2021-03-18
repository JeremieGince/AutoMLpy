import warnings
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from typing import Dict, Union, List, Iterable, Hashable, Tuple, Callable
import pandas as pd
from multiprocessing import Pool
import time
import os
import logging
import plotly.graph_objects as go
import json
import warnings

# ------- App Server -------- #
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import threading
import plotly.io as pio


class ParameterGenerator:

    class Decorators:
        @classmethod
        def increment_counters(cls, method):
            def wrapper(self, *args, **kwargs):
                out = method(self, *args, **kwargs)
                self.elapse_time_per_iteration[self.current_itr] = self.elapse_time
                self.current_itr += 1
                return out

            return wrapper

    def __init__(self,
                 values_dict: Dict[Union[int, str], Iterable[Union[int, float]]],
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

        }

        Attributes
        ----------
        self._param_name_to_idx (Dict[str, int]) : Dict container for string bounds element used to convert str to idx.
        self._param_idx_to_name (Dict[int, str]) : Dict container for string bounds element used to convert back idx
                                                   to str.
        self._param_name_to_type (Dict[Union[str, int], Callable]): Conversion table to convert param to
                                                                    it's initial type.
        self.current_itr (int): Current iteration of the gpo.
        self.start_time (int): Starting time of the optimisation.
        self.history List[Tuple]: The history of the hp search.
        """
        # ------- Conversion tables -------- #
        self._param_name_to_idx: Dict[str, Dict[str, int]] = {}
        self._param_idx_to_name: Dict[str, Dict[int, str]] = {}
        self._param_name_to_type: Dict[Union[str, int], Callable] = {}

        self._values_names = list(values_dict.keys())
        self._values_dict = values_dict

        for p in self._values_names:
            self._param_name_to_type[p] = type(self._values_dict[p][0])
            if self.check_str_in_iterable(values_dict[p]):
                self.add_conversion_tables_param_name_to_idx(p, values_dict[p])
                values_dict[p] = self.convert_param_to_idx(p, values_dict[p])

        # ------- Hp score_space -------- #
        self.xx = np.meshgrid(*[values_dict[p] for p in self._values_names])
        self.xx = np.array(list(zip(*[_x.ravel() for _x in self.xx])))

        # ------- Counters -------- #
        self.current_itr: int = 0
        self.max_itr = int(kwargs.get("max_itr", len(self.xx)))
        self.start_time = time.time()
        self.max_seconds = kwargs.get("max_seconds", 60 ** 2)
        self.elapse_time_per_iteration: dict = {}

        # ------- History containers -------- #
        self.history = []

        # ------- App Server -------- #
        self.app = None
        self.app_thread = None

        # ------- Saving -------- #
        self.default_save_dir = kwargs.get("save_dir", f"parameter_generators_data/{self.__class__.__name__}/")
        self.default_save_name = kwargs.get("save_name", '-'.join(self._values_names))

    def __del__(self):
        self.close_graph_server()

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

    @property
    def last_itr_elapse_time(self) -> float:
        return self.elapse_time_per_iteration[self.current_itr-1]

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

    def __eq__(self, other):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.__dict__)

    @Decorators.increment_counters
    def get_trial_param(self) -> Dict[str, Union[int, float]]:
        """
        Returned a set of trial parameter.
        """
        raise NotImplementedError()

    def get_best_param(self, **kwargs) -> Dict[str, Union[int, float]]:
        """
        Get the best predicted parameters with the current exploration.
        """
        if len(self.history) == 0:
            raise ValueError("get_best_param must be called after an optimisation")
        return max(self.history, key=lambda t: t[-1])[0]

    def get_best_params_repr(self, **kwargs) -> str:
        if len(self.history) > 0:
            predicted_best_param_repr = ""
            for k, v in self.get_best_param(**kwargs).items():
                if isinstance(v, float):
                    predicted_best_param_repr += f"\t{k}: {v:.3f}\n"
                else:
                    predicted_best_param_repr += f"\t{k}: {v}\n"
        else:
            predicted_best_param_repr = "None"
        return predicted_best_param_repr

    def add_score_info(self, param: Dict[str, Union[int, float]], score: float) -> None:
        """
        Add the result of the trial parameters.

        Parameters
        ----------
        param: The trial parameters.
        score: The associated score of the trial parameters.
        """
        self.history.append((param, score))
        # for p_name in param:
        #     if p_name in self._param_name_to_idx:
        #         param[p_name] = self._param_name_to_idx[p_name][param[p_name]]

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

    def convert_idx_to_param(self, param_name: str, indexes: Iterable[int]) -> List[str]:
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

    def convert_subspace_to_param(self, sub_space: np.ndarray) -> Dict[Union[str, int], object]:
        """
        Convert a subspace of score_space self.xx to a set of parameters.
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

        _param = {p_name: self._param_name_to_type[p_name](v) for p_name, v in _param.items()}
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
        Show the expectation of hp-score_space.
        """
        logging.error(DeprecationWarning("Use write_optimization_to_html instead"))
        pass

    def start_graph_server(self, **kwargs):
        logging.error(DeprecationWarning("Use write_optimization_to_html instead"))

        if self.app_thread is not None:
            return

        self.app = dash.Dash(__name__)

        self.app.layout = html.Div([
            html.P("Hyper-parameter: "),
            dcc.Dropdown(
                id="dropdown",
                options=[
                    {'label': p_name, 'value': p_name}
                    for p_name in self._values_names
                ],
                value=self._values_names[0],
                clearable=False,
            ),
            dcc.Graph(id="hp-graph"),
        ])

        self.app.callback(Output("hp-graph", "figure"), [Input("dropdown", "value")])(self.update_graph_server)
        self.app.run_server(
            debug=True,
            port=kwargs.get("port", 80),
            host=kwargs.get("host", "127.0.0.1"),
            use_reloader=False,
        )
        # self.app_thread = threading.Thread(target=self.app.run_server,
        #                                    kwargs=dict(
        #                                        debug=True,
        #                                        port=kwargs.get("port", 80),
        #                                        host=kwargs.get("host", "127.0.0.1")
        #                                    ))
        # self.app_thread.start()

    def close_graph_server(self):
        if self.app_thread is None:
            return
        self.app_thread.join()
        self.app_thread = None

    def update_graph_server(self, parameter_name: str):
        param_trial_x = []
        param_trial_score = []
        for i, (p_trial, p_score) in enumerate(self.history):
            param_trial_x.append(p_trial[parameter_name])
            param_trial_score.append(p_score)

        x, y = np.array(param_trial_x), np.array(param_trial_score)

        fig = go.Figure(
            data=[
                go.Scatter(x=x, y=y,
                           mode='markers',
                           name="Trial points",
                           marker=dict(size=5, color=y, colorscale='Viridis', showscale=True),
                           ),
            ]
        )

        fig.update_layout(
            width=800,
            height=900,
            autosize=False,
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_white",
        )

        fig.update_xaxes(title=f"{parameter_name}: parameter space [-]")
        fig.update_yaxes(title="Score [-]")
        return fig

    def _compute_x_y_dict(self, **kwargs) -> Dict:
        x_y_dict = {p_name: dict() for p_name in self._values_names}

        for p_name in x_y_dict:
            param_trial_x = []
            param_trial_score = []
            for i, (p_trial, p_score) in enumerate(self.history):
                param_trial_x.append(p_trial[p_name])
                param_trial_score.append(p_score)

            x, y = np.array(param_trial_x), np.array(param_trial_score)
            x_y_dict[p_name] = dict(x=x, y=y)
        return x_y_dict

    def _init_html_fig(self, x_y_dict: Dict, **kwargs) -> go.Figure:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=x_y_dict[self._values_names[0]]['x'],
                       y=x_y_dict[self._values_names[0]]['y'],
                       mode='markers',
                       name="Trial points",
                       marker=dict(size=5, color=x_y_dict[self._values_names[0]]['y'],
                                   colorscale='Viridis', showscale=True),
                       ),
        )

        fig.update_xaxes(title=f"{self._values_names[0]}: parameter space [-]")
        fig.update_yaxes(title="Score [-]")

        fig.update_layout(
            title=kwargs.get("title", ""),
            # width=1080,
            # height=750,
            autosize=True,
            margin=dict(t=150, b=150, l=150, r=150),
            template="plotly_dark" if kwargs.get("dark_mode", True) else "seaborn",
            font=dict(
                size=18,
            )
        )

        # Update 3D scene options
        fig.update_scenes(
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode="manual"
        )

        return fig

    def _add_dropdown_html_fig_(self, fig, x_y_dict, **kwargs):
        # Add dropdown
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[
                                dict(
                                    x=[x_y_dict[p_name]['x']],
                                    y=[x_y_dict[p_name]['y']],
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

    def _add_annotations_to_html_fig_(self, fig, x_y_dict, **kwargs):
        annotations = [
            dict(text="Hyper-parameter:", showarrow=False,
                 x=0.89, y=1.1, xref="paper", yref="paper", align="left",
                 xanchor="right", yanchor="middle")
        ]
        if kwargs.get("add_best_hp_annotation", False):
            annotations.append(
                dict(text=f"Predicted best hyper-parameters:"
                          f" {self.get_best_params_repr(**kwargs.get('get_best_param_kwargs', {}))}",
                     showarrow=False,
                     x=0.1, y=-0.1, xref="paper", yref="paper", align="left",
                     xanchor="left", yanchor="top")
            )
        fig.update_layout(annotations=annotations)

    def write_optimization_to_html(self, **kwargs) -> go.Figure:
        """
        Write the optimization search in an interactive html file.

        Parameters
        ----------
        kwargs:
            add_best_hp_annotation (bool): True to add a annotation that show the predicted best hyper-parameters.
             Default False.

        Returns
        -------
        The html figure.
        """
        x_y_dict = self._compute_x_y_dict(**kwargs)
        fig = self._init_html_fig(x_y_dict, **kwargs)

        self._add_dropdown_html_fig_(fig, x_y_dict, **kwargs)
        self._add_annotations_to_html_fig_(fig, x_y_dict, **kwargs)

        self.save_html_fig(fig, **kwargs)
        if kwargs.get("show", True):
            fig.show()

        return fig

    def save_html_fig(self, fig, **kwargs):
        save_dir = kwargs.get("save_dir", f"{self.default_save_dir}/html_files/")
        os.makedirs(save_dir, exist_ok=True)

        path = f"{save_dir}/{self.default_save_name}-{kwargs.get('save_name', '')}.html"
        if kwargs.get("save", True):
            logging.info(f"Saving html fig to {path}")
            fig.write_html(path)

    def save_best_param(self, **kwargs):
        save_dir = kwargs.get("save_dir", f"{self.default_save_dir}/optimal_hp/")
        save_name = kwargs.get("save_name", f"{self.default_save_name}-opt_hp")
        os.makedirs(save_dir, exist_ok=True)
        # save_path = save_dir + '/' + save_name
        # np.save(save_path + ".npy", self.get_best_param(), allow_pickle=True)

        with open(f'{save_dir}/{save_name}.json', 'w') as f:
            json.dump(self.get_best_param(), f, indent=4)

    def save_obj(self):
        raise NotImplementedError()

    @staticmethod
    def load_obj(path: str):
        raise NotImplementedError()
