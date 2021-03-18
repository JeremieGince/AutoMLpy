from tests.objective_functions.vectorized_objective_function import VectorizedObjectiveFuncHpOptimizer
import numpy as np
from src.parameter_generators import SearchType, GridHpSearch, RandomHpSearch, search_type_2_type
from src import logs_file_setup, log_device_setup, DeepLib
from src import HpOptimizer


def execute_optimisation(
        search_type: SearchType,
        dim: int = 3,
        nb_workers: int = 1,
        compute_delay: float = 0.0,
        param_search_kwargs: dict = None,
        optimize_kwargs: dict = None,
        **kwargs
):
    if param_search_kwargs is None:
        param_search_kwargs = dict(
            max_seconds=60 * 60 * 1
        )
    if optimize_kwargs is None:
        optimize_kwargs = {}

    seed = kwargs.get("seed", 42)
    np.random.seed(seed)

    obj_func_hp_optimizer = VectorizedObjectiveFuncHpOptimizer()
    obj_func_hp_optimizer.set_dim(dim)
    obj_func_hp_optimizer.set_compute_delay(compute_delay)
    _ = obj_func_hp_optimizer.build_model()

    param_gen = search_type_2_type[search_type](
        obj_func_hp_optimizer.hp_space,
        **param_search_kwargs
    )

    save_kwargs = dict(
        save_name=f"vec_{dim}D_obj_func_hp_opt_seed{seed}",
        title=f"{search_type.name} search: {dim}D vectorized objective function",
        dark_mode=kwargs.get("dark_mode", True),
    )

    param_gen = obj_func_hp_optimizer.optimize(
        param_gen,
        np.ones((2, 2)),
        np.ones((2, 2)),
        n_splits=2,
        nb_workers=nb_workers,
        save_kwargs=save_kwargs,
        **optimize_kwargs,
    )

    opt_hp = param_gen.get_best_param()

    test_acc, _ = obj_func_hp_optimizer.score(obj_func_hp_optimizer.build_model(**opt_hp), **opt_hp)

    param_gen.write_optimization_to_html(show=kwargs.get("show", False), **save_kwargs)
    return param_gen
