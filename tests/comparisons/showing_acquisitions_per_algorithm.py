from src.parameter_generators import SearchType
from tests import execute_optimisation


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
        for _search_type in SearchType:
            param_gen = execute_optimisation(
                _search_type,
                dim=d,
                param_search_kwargs=param_search_kwargs,
                optimize_kwargs=optimize_kwargs,
                seed=seed,
                show=kwargs.get("show", True),
            )


if __name__ == '__main__':
    from src import logs_file_setup

    logs_file_setup(__file__)
    show_optimisations([1, ], stop_criterion=None, max_itr=1000, max_seconds=60*60)
