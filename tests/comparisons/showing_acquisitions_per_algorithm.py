from src.parameter_generators import SearchType
from tests import execute_optimisation
import plotly.graph_objects as go
from tests.objective_functions.vectorized_objective_function import VectorizedObjectiveFuncHpOptimizer
import os


def show_1d_vectorized_objective_function(**kwargs):
    obj_func_hp_optimizer = VectorizedObjectiveFuncHpOptimizer()
    obj_func_hp_optimizer.set_dim(1)
    func = obj_func_hp_optimizer.build_model()
    Z, _ = obj_func_hp_optimizer.score(
        func,
        **{f"x{i}": obj_func_hp_optimizer.meshgrid[i] for i in range(obj_func_hp_optimizer.dim)}
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=obj_func_hp_optimizer.hp_space["x0"],
                   y=Z,
                   mode='lines',),
    )
    fig.update_xaxes(title=f"x0: parameter space [-]")
    fig.update_yaxes(title="Score [-]")
    fig.update_layout(
        title=f'{obj_func_hp_optimizer.dim}D vectorized objective function',
        autosize=True,
        margin=dict(t=150, b=150, l=150, r=150),
        template="plotly_dark" if kwargs.get("dark_mode", True) else "seaborn",
        font=dict(
            size=18,
        )
    )
    save_dir = f"figures/{obj_func_hp_optimizer.dim}D_vectorized_objective_function/"
    os.makedirs(save_dir, exist_ok=True)
    fig.write_html(f"{save_dir}/{'-'.join([f'{k}_{v}' for k, v in obj_func_hp_optimizer.params.items()])}.html")
    fig.show()


def show_optimisations(
        dims: list,
        seed: int = 42,
        param_search_kwargs: dict = None,
        optimize_kwargs: dict = None,
        **kwargs
):
    if param_search_kwargs is None:
        param_search_kwargs = dict(
            max_seconds=kwargs.get("max_seconds", 60 * 60 * 1),
            max_itr=kwargs.get("max_itr", 500_000),
        )
    if optimize_kwargs is None:
        optimize_kwargs = dict(
            stop_criterion=kwargs.get("stop_criterion", 0.9)
        )

    for d in dims:
        if d == 1:
            show_1d_vectorized_objective_function(**kwargs)
        for _search_type in SearchType:
            param_gen = execute_optimisation(
                _search_type,
                dim=d,
                param_search_kwargs=param_search_kwargs,
                optimize_kwargs=optimize_kwargs,
                seed=seed,
                show=kwargs.get("show", True),
                **kwargs
            )


if __name__ == '__main__':
    from src import logs_file_setup

    logs_file_setup(__file__)
    show_optimisations([1, ], stop_criterion=None, max_itr=1000, max_seconds=60*60, dark_mode=False)
