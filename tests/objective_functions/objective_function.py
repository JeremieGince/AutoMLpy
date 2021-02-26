from typing import Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np

from src.optimizer import HpOptimizer


def objective_function(x, y, **kwargs):
    A = kwargs.get("A", 3)
    x_0 = kwargs.get("x_0", 0.1)
    sigma_x = kwargs.get("sigma_x", 0.5)
    y_0 = kwargs.get("y_0", 0.1)
    sigma_y = kwargs.get("sigma_y", 0.5)
    return A * np.sin(np.sqrt(((x - x_0) ** 2) / (2 * sigma_x ** 2) + ((y - y_0) ** 2) / (2 * sigma_y ** 2)))


def vectorized_objective_function(x, **kwargs):
    beta = kwargs.get("beta", np.random.normal(kwargs.get("beta_loc", 0.0), kwargs.get("beta_scale", 0.1)))
    beta = np.clip(beta, 1e-12, 1.0)
    
    sigma = kwargs.get("sigma", np.random.normal(kwargs.get("sigma_loc", 0.0), kwargs.get("sigma_scale", 0.5)))
    sigma = np.clip(sigma, 1e-12, 1.0)
    
    A = kwargs.get("A", 1.0)
    A = np.clip(A, 1e-12, 1.0)
    
    noise = kwargs.get("noise", 0.0) * np.random.normal(0.0, 1.0)
    return noise * A * np.sin(np.sqrt(np.sum(((x - beta) ** 2) / (2 * sigma ** 2), axis=0)))


class ObjectiveFuncHpOptimizer(HpOptimizer):
    hp_space = dict(
        x0=np.linspace(0, 1, 1_000),
        x1=np.linspace(0, 1, 1_000),
    )
    params = dict(
        A=1,
        x_0=0.1,
        sigma_x=0.3,
        y_0=0.1,
        sigma_y=0.5,
    )
    meshgrid = None
    max: float = 1.0

    def build_model(self, **hp) -> Callable:
        if self.meshgrid is None:
            self.meshgrid = np.meshgrid(self.hp_space["x0"], self.hp_space["x1"])
            self.max = np.max(objective_function(*self.meshgrid, **self.params))
        return objective_function

    def fit_model_(self,
                   model,
                   X: np.ndarray = None,
                   y: np.array = None,
                   **hp
                   ) -> None:
        pass

    def score(self,
              model: Callable,
              X: np.ndarray = None,
              y: np.ndarray = None,
              **hp
              ) -> Tuple[float, float]:
        z = model(hp.get("x0"), hp.get("x1"), **self.params)
        test_loss, test_acc = (1.0 - z / self.max) ** 2, z
        return test_acc, 0.0


class VectorizedObjectiveFuncHpOptimizer(HpOptimizer):
    dim = 2
    hp_space = {
        f"x{i}": np.linspace(0, 1, 1_000) for i in range(dim)
    }
    params = dict(
        # A=1,
        # beta=np.random.normal(0.0, 0.1),
        # sigma=np.random.normal(0.0, 0.5),
        # noise=0.1,
        A=3,
        beta=np.array([0.1 for _ in range(dim)]),
        sigma=np.array([0.5 for _ in range(dim)]),
        noise=0.0,
    )
    meshgrid = None
    max: float = 1.0

    def build_model(self, **hp) -> Callable:
        if self.meshgrid is None:
            self.meshgrid = np.meshgrid(*list(self.hp_space.values()))
            self.max = np.max(vectorized_objective_function(self.meshgrid, **self.params))
        return vectorized_objective_function

    def fit_model_(self,
                   model,
                   X: np.ndarray = None,
                   y: np.array = None,
                   **hp
                   ) -> None:
        pass

    def score(self,
              model: Callable,
              X: np.ndarray = None,
              y: np.ndarray = None,
              **hp
              ) -> Tuple[float, float]:
        z = model(np.array([hp[f"x{i}"] for i in range(self.dim)]), **self.params)
        test_loss, test_acc = (1.0 - z / self.max) ** 2, z
        return test_acc, 0.0


def visualize_objective_function():
    import plotly.graph_objects as go
    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    import time
    from src.random_search import RandomHpSearch
    from src.grid_search import GridHpSearch
    import os

    # ----------------- Initialization -------------------- #
    obj_func_hp_optimizer = ObjectiveFuncHpOptimizer()
    func = obj_func_hp_optimizer.build_model()
    print(f"Maximum of objective function: {obj_func_hp_optimizer.max:.3f}")
    Z, _ = obj_func_hp_optimizer.score(func, x0=obj_func_hp_optimizer.meshgrid[0], x1=obj_func_hp_optimizer.meshgrid[1])

    # ----------------- Figure -------------------- #
    fig = go.Figure(data=[go.Surface(x=obj_func_hp_optimizer.hp_space["x0"],
                                     y=obj_func_hp_optimizer.hp_space["x1"],
                                     z=Z)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_layout(title='Objective function', autosize=True)

    save_dir = f"../figures/objective_function/"
    os.makedirs(save_dir, exist_ok=True)
    fig.write_html(f"{save_dir}/{'-'.join([f'{k}_{v}' for k, v in obj_func_hp_optimizer.params.items()])}.html")
    fig.show()

    # ----------------- Web Server -------------------- #
    # app = dash.Dash()
    # app.layout = html.Div([
    #     dcc.Graph(figure=fig)
    # ])
    # app.run_server(debug=True, use_reloader=False)

    # ----------------- Optimization -------------------- #
    start_time = time.time()

    param_gen = RandomHpSearch(obj_func_hp_optimizer.hp_space, max_itr=10_000, max_seconds=60)
    # param_gen.start_graph_server()

    param_gen.get_trial_param()

    opt_param_gen = obj_func_hp_optimizer.optimize(
        param_gen,
        np.ones((2, 2)),
        np.ones((2, 2)),
        n_splits=2
    )

    opt_hp = opt_param_gen.get_best_param()
    test_acc, _ = obj_func_hp_optimizer.score(obj_func_hp_optimizer.build_model(), x0=opt_hp["x0"], x1=opt_hp["x1"])

    assert test_acc >= 0.95, f"Random Gen result: {test_acc:.2f}% in {time.time() - start_time:.2f} [s]"
    print(f"{opt_hp}, test_acc: {test_acc}")

    opt_param_gen.write_optimization_to_html()


def visualize_vectorized_objective_function():
    import plotly.graph_objects as go
    from src.gp_search import GPOHpSearch
    import os
    import time

    # ----------------- Initialization -------------------- #
    obj_func_hp_optimizer = VectorizedObjectiveFuncHpOptimizer()
    func = obj_func_hp_optimizer.build_model()
    print(f"Maximum of objective function: {obj_func_hp_optimizer.max:.3f}")
    Z, _ = obj_func_hp_optimizer.score(
        func,
        **{f"x{i}": obj_func_hp_optimizer.meshgrid[i] for i in range(obj_func_hp_optimizer.dim)}
    )

    # ----------------- Figure -------------------- #
    fig = go.Figure(data=[go.Surface(x=obj_func_hp_optimizer.hp_space["x0"],
                                     y=obj_func_hp_optimizer.hp_space["x1"],
                                     z=Z)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_layout(title='Vectorized objective function', autosize=True)

    save_dir = f"../figures/vectorized_objective_function/"
    os.makedirs(save_dir, exist_ok=True)
    fig.write_html(f"{save_dir}/{'-'.join([f'{k}_{v}' for k, v in obj_func_hp_optimizer.params.items()])}.html")
    fig.show()

    # ----------------- Optimization -------------------- #
    start_time = time.time()

    param_gen = GPOHpSearch(obj_func_hp_optimizer.hp_space, max_itr=10_000, max_seconds=60)
    # param_gen.start_graph_server()

    param_gen.get_trial_param()

    opt_param_gen = obj_func_hp_optimizer.optimize(
        param_gen,
        np.ones((2, 2)),
        np.ones((2, 2)),
        n_splits=2
    )

    opt_hp = opt_param_gen.get_best_param()
    test_acc, _ = obj_func_hp_optimizer.score(
        obj_func_hp_optimizer.build_model(),
        **opt_hp
    )

    assert test_acc >= 0.95, f"GPO Gen result: {test_acc:.2f}% in {time.time() - start_time:.2f} [s]"
    print(f"{opt_hp}, test_acc: {test_acc}")

    opt_param_gen.write_optimization_to_html()


if __name__ == '__main__':
    visualize_vectorized_objective_function()
