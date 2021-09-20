from typing import Callable
import os
import time
import numpy as np

from src.AutoMLpy.optimizers.optimizer import HpOptimizer


def vectorized_objective_function__(x, **kwargs):
    beta = kwargs.get("beta", np.random.normal(kwargs.get("beta_loc", 0.0), kwargs.get("beta_scale", 0.1)))
    beta = np.clip(beta, 1e-12, 1.0)

    sigma = kwargs.get("sigma", np.random.normal(kwargs.get("sigma_loc", 0.0), kwargs.get("sigma_scale", 0.5)))
    sigma = np.clip(sigma, 1e-12, 1.0)

    A = kwargs.get("A", 1.0)
    A = np.clip(A, 1e-12, 1.0)

    noise = kwargs.get("noise", 0.0) * np.random.normal(0.0, 1.0)
    return A * np.sin(np.sqrt(np.sum(((x - beta) ** 2) / (2 * sigma ** 2), axis=0))) + noise


def vectorized_objective_function(x, **kwargs):
    omega = kwargs.get("omega", np.ones(x.shape))
    assert omega.shape[0] == x.shape[0]
    noise = kwargs.get("noise", 0.0) * np.random.normal(0.0, 1.0)

    omega_x = np.array([omega[i]*x[i] for i in range(x.shape[0])])
    omega_xx = np.array([omega[i]*x[i]**2 for i in range(x.shape[0])])

    cos_prod = np.prod(np.cos(omega_x), axis=0)
    exp_prod = np.exp(-np.sum(omega_xx, axis=0))

    time.sleep(kwargs.get("compute_delay", 0.0))
    return cos_prod*exp_prod + noise


class VectorizedObjectiveFuncHpOptimizer(HpOptimizer):
    dim = 2
    nb_points = 100_000
    hp_space = None
    params = None
    meshgrid = None
    max: float = 1.0
    min: float = 0.0
    score_space = None
    compute_delay: float = 0.0  # [secs]

    def set_compute_delay(self, new_compute_delay: float):
        self.compute_delay = new_compute_delay

    def set_dim(self, new_dim: int):
        self.dim = new_dim

    def build_model(self, **hp) -> Callable:
        if self.meshgrid is None:
            self.hp_space = {
                f"x{i}": np.linspace(-1, 1, int(np.floor(self.nb_points**(1/self.dim)))) for i in range(self.dim)
            }
            self.params = dict(
                # omega=np.random.normal(10.0, 1.0, self.dim),
                omega=np.array([1*10**(i/self.dim+1) for i in range(self.dim)]),
                noise=0.0,
                compute_delay=self.compute_delay,
            )
            self.meshgrid = np.meshgrid(*[self.hp_space[f"x{i}"] for i in range(self.dim)])
            self.score_space = vectorized_objective_function(np.array(self.meshgrid), **self.params)
            self.max = np.max(self.score_space)
            self.min = np.min(self.score_space)
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
              ) -> float:
        z = model(np.array([hp[f"x{i}"] for i in range(self.dim)]), **self.params)
        return (z - self.min) / (self.max - self.min)


if __name__ == '__main__':
    import plotly.graph_objects as go
    from src.AutoMLpy import GPOHpSearch

    # ----------------- Initialization -------------------- #
    obj_func_hp_optimizer = VectorizedObjectiveFuncHpOptimizer()
    obj_func_hp_optimizer.nb_points = 1_000
    obj_func_hp_optimizer.set_dim(1)
    func = obj_func_hp_optimizer.build_model()
    print(f"Maximum of objective function: {obj_func_hp_optimizer.max:.3f}")
    # Z = obj_func_hp_optimizer.score(
    #     func,
    #     **{f"x{i}": obj_func_hp_optimizer.meshgrid[i] for i in range(obj_func_hp_optimizer.dim)}
    # )
    #
    # # print(f"Z.shape: {Z.shape}")
    # # print(f"Z[0, 1].shape: {Z[:, :, 0].shape}")
    #
    # # ----------------- Figure -------------------- #
    # fig = go.Figure()
    #
    # fig.add_trace(
    #     go.Surface(
    #         x=obj_func_hp_optimizer.hp_space["x0"],
    #         y=obj_func_hp_optimizer.hp_space["x1"],
    #         z=Z[:, :, ],
    #     )
    # )
    # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
    #                                   highlightcolor="limegreen", project_z=True))
    # fig.update_layout(
    #     title=f'{obj_func_hp_optimizer.dim}D vectorized objective function',
    #     autosize=True,
    #     margin=dict(t=150, b=150, l=150, r=150),
    #     template="seaborn",
    # )
    # fig.update_xaxes(title=f"x0: parameter space [-]")
    # fig.update_yaxes(title=f"x1: parameter space [-]")
    # # fig.update_zaxes(title="Score [-]")
    #
    # # Add dropdown
    # fig.update_layout(
    #     updatemenus=[
    #         dict(
    #             buttons=list([
    #                 dict(
    #                     args=[
    #                         dict(
    #                             x=[obj_func_hp_optimizer.hp_space[xi]],
    #                             y=[obj_func_hp_optimizer.hp_space[xj]],
    #                             z=Z,
    #                         ),
    #                         {
    #                             # "title": f"{xi}-{xj}",
    #                             "xaxis.title.text": f"{xi}: parameter space [-]",
    #                             "yaxis.title.text": f"{xj}: parameter space [-]",
    #                             "zaxis.title.text": "Score [-]",
    #                         }
    #                     ],
    #                     label=f"{xi}-{xj}",
    #                     method="update",
    #                 )
    #                 for xi in obj_func_hp_optimizer.hp_space for xj in obj_func_hp_optimizer.hp_space if xi != xj
    #             ]),
    #             direction="down",
    #             pad={"r": 10, "t": 10},
    #             showactive=True,
    #             x=0.9,
    #             xanchor="left",
    #             y=1.1,
    #             yanchor="middle"
    #         ),
    #     ]
    # )
    #
    # save_dir = f"figures/vectorized_objective_function/"
    # os.makedirs(save_dir, exist_ok=True)
    # fig.write_html(f"{save_dir}/{'-'.join([f'{k}_{v}' for k, v in obj_func_hp_optimizer.params.items()])}.html")
    # fig.show()

    # ----------------- Optimization -------------------- #
    start_time = time.time()

    param_gen = GPOHpSearch(
        obj_func_hp_optimizer.hp_space,
        # max_seconds=60*1,
        max_itr=10,
        xi=0.1,
    )

    opt_param_gen = obj_func_hp_optimizer.optimize(
        param_gen,
        np.ones((2, 2)),
        np.ones((2, 2)),
        n_splits=2,
        # stop_criterion=0.95,
    )

    opt_hp = opt_param_gen.get_best_param(from_history=False)
    test_acc = obj_func_hp_optimizer.score(
        obj_func_hp_optimizer.build_model(),
        **opt_hp
    )

    opt_param_gen.write_optimization_to_html(dark_mode=False)
    assert test_acc >= 0.95, f"GPO Gen result: {test_acc*100:.3f}% in {time.time() - start_time:.2f} [s]"
    print(f"{opt_hp}, test_acc: {test_acc*100:.3f}")

