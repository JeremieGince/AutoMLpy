from src.AutoMLpy.parameter_generators import SearchType
from src.AutoMLpy import logs_file_setup, log_device_setup
from tests import execute_optimisation
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from src.AutoMLpy.tools import plotly_colors
import logging
import multiprocessing
import tracemalloc

try:
    import tensorflow as tf
    tf.get_logger().setLevel(logging.FATAL)
except ImportError:
    pass


def compute_stats_per_workers_table(
        max_workers: int = 10,
        dim: int = 1,
        iterations_per_workers: int = 10,
        compute_delay: float = 1.0,
        **kwargs
):
    algo_types = kwargs.get("algos", SearchType)
    columns = ["Workers", *[st.name for st in algo_types]]

    iterations_results = pd.DataFrame(columns=columns)
    time_results = pd.DataFrame(columns=columns)
    memory_results = pd.DataFrame(columns=columns)

    for w in range(1, max_workers+1):
        logging.info(f"\n{'-'*50} {w} Workers {'-'*50}")
        new_iterations_results = {"Workers": w, **{st.name: [] for st in algo_types}}
        new_time_results = {"Workers": w, **{st.name: [] for st in algo_types}}
        new_memory_results = {"Workers": w, **{st.name: [] for st in algo_types}}

        for _search_type in algo_types:
            logging.info(f"\n{'-'*10} {_search_type.name} search {'-'*10}")
            ell_itr, ell_time, ell_mem = [], [], []
            for itr_seed in range(iterations_per_workers):
                tracemalloc.start()
                param_gen = execute_optimisation(
                    _search_type,
                    dim=dim,
                    nb_workers=w,
                    compute_delay=compute_delay,
                    optimize_kwargs=dict(stop_criterion=kwargs.get("stop_criterion", 0.9)),
                    seed=itr_seed,
                    **kwargs
                )
                current_mem, peak_mem = tracemalloc.get_traced_memory()

                ell_itr.append(param_gen.current_itr)
                ell_time.append(param_gen.last_itr_elapse_time)
                ell_mem.append(peak_mem * 1e-6)  # convert bytes to MB

                tracemalloc.stop()

            new_iterations_results[_search_type.name] = (np.mean(ell_itr), np.std(ell_itr))
            new_time_results[_search_type.name] = (np.mean(ell_time), np.std(ell_time))
            new_memory_results[_search_type.name] = (np.mean(ell_mem), np.std(ell_mem))

        iterations_results = iterations_results.append(new_iterations_results, ignore_index=True)
        time_results = time_results.append(new_time_results, ignore_index=True)
        memory_results = memory_results.append(new_memory_results, ignore_index=True)

    return iterations_results, time_results, memory_results


def show_stats_per_dimension(
        max_workers: int = 10,
        dim: int = 1,
        iterations_per_workers: int = 10,
        compute_delay: float = 1.0,
        **kwargs
):

    iterations_results, time_results, memory_results = compute_stats_per_workers_table(
        max_workers, dim, iterations_per_workers, compute_delay, **kwargs
    )
    keys = [st.name for st in kwargs.get("algos", SearchType)]

    iterations_results_mean = {
        st: np.array([x[0] for x in iterations_results[st]])
        for st in keys
    }
    iterations_results_std = {
        st: np.array([x[1] for x in iterations_results[st]])
        for st in keys
    }

    time_results_mean = {
        st: np.array([x[0] for x in time_results[st]])
        for st in keys
    }
    time_results_std = {
        st: np.array([x[1] for x in time_results[st]])
        for st in keys
    }

    memory_results_mean = {
        st: np.array([x[0] for x in memory_results[st]])
        for st in keys
    }
    memory_results_std = {
        st: np.array([x[1] for x in memory_results[st]])
        for st in keys
    }

    iterations_y_list = []
    time_y_list = []
    memory_y_list = []

    # --------------------------------------------------------------------------------- #
    #                              Initialize figure                                    #
    # --------------------------------------------------------------------------------- #

    fig = go.Figure()

    for i, st in enumerate(keys):
        x = list(iterations_results["Workers"])

        itr_std_upper = list(iterations_results_mean[st] + iterations_results_std[st])
        itr_std_lower = list(iterations_results_mean[st] - iterations_results_std[st])
        itr_mean = list(iterations_results_mean[st])
        itr_std = itr_std_lower + itr_std_upper[::-1]

        time_std_upper = list(time_results_mean[st] + time_results_std[st])
        time_std_lower = list(time_results_mean[st] - time_results_std[st])
        time_mean = list(time_results_mean[st])
        time_std = time_std_lower + time_std_upper[::-1]

        memory_std_upper = list(memory_results_mean[st] + memory_results_std[st])
        memory_std_lower = list(memory_results_mean[st] - memory_results_std[st])
        memory_mean = list(memory_results_mean[st])
        memory_std = memory_std_lower + memory_std_upper[::-1]

        fig.add_trace(
            go.Scatter(x=x,
                       y=itr_mean,
                       mode='lines',
                       name=f"{st} Search mean",
                       line_color=plotly_colors[i], )
        )
        fig.add_trace(
            go.Scatter(x=x+x[::-1],
                       y=itr_std,
                       mode='lines',
                       fill="toself",
                       fillcolor=plotly_colors[i],
                       name=f"{st} Search std",
                       line=dict(width=0.0),
                       opacity=0.5,)
        )

        iterations_y_list.append(itr_mean)
        iterations_y_list.append(itr_std)

        time_y_list.append(time_mean)
        time_y_list.append(time_std)

        memory_y_list.append(memory_mean)
        memory_y_list.append(memory_std)

    fig.update_xaxes(title=f"Workers")
    fig.update_yaxes(title="Iterations [-]")

    fig.update_layout(
        title=kwargs.get("title", f"Iterations required to obtain a score of {kwargs['stop_criterion']}"),
        autosize=True,
        margin=dict(t=150, b=150, l=150, r=150),
        template="plotly_dark" if kwargs.get("dark_mode", True) else "seaborn",
        font=dict(
            size=18,
        )
    )

    # --------------------------------------------------------------------------------- #
    #                                 Add Dropdown                                      #
    # --------------------------------------------------------------------------------- #
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[
                            dict(
                                y=y_list,
                            ),
                            {
                                "title": f"{' '.join(label.split(' ')[:-1])} "
                                         f"required to obtain a score of {kwargs['stop_criterion']}",
                                "xaxis.title.text": f"Workers [-]",
                                "yaxis.title.text": f"{label}",
                            }
                        ],
                        label=f"yaxis: {label}",
                        method="update"
                    )
                    for y_list, label in zip(
                        [iterations_y_list, time_y_list, memory_y_list],
                        ["Iterations [-]", "Time [s]", "Memory [MB]"]
                    )
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

    # --------------------------------------------------------------------------------- #
    #                               Saving/showing                                      #
    # --------------------------------------------------------------------------------- #
    save_dir = kwargs.get("save_dir", f"figures/html_files/")
    os.makedirs(save_dir, exist_ok=True)
    if kwargs.get("save", True):
        fig.write_html(f"{save_dir}/algorithms_workers_comparison-algos[{'-'.join(keys)}]"
                       f"-maxworkers{max_workers}-dim{dim}-iteration{iterations_per_workers}.html")

    fig.show()
    return iterations_results, time_results, memory_results


if __name__ == '__main__':
    logs_file_setup(__file__, level=logging.INFO)
    log_device_setup()

    iterations_results, time_results, memory_results = show_stats_per_dimension(
        max_workers=min(5, multiprocessing.cpu_count()//2),
        dim=1,
        iterations_per_workers=10,
        compute_delay=0.05,
        stop_criterion=0.75,
        algos=[SearchType.Random, SearchType.GPO, ],
        dark_mode=False
    )

    # --------------------------------------------------------------------------------- #
    #                      Iteration per Workers results                                #
    # --------------------------------------------------------------------------------- #

    logging.info('\n' + ('-' * 125) + '\n' + "Iteration per Workers results" + '\n' + ('-' * 125))
    logging.info(iterations_results)
    logging.info(('-' * 125) + '\n')

    logging.info('\n' + ('-' * 125) + '\n' + "Iteration per Workers results LaTex" + '\n' + ('-' * 125))
    logging.info(iterations_results.to_latex())
    logging.info(('-' * 125) + '\n')

    # --------------------------------------------------------------------------------- #
    #                             Time per Workers results                              #
    # --------------------------------------------------------------------------------- #

    logging.info('\n' + ('-' * 125) + '\n' + "Time per Workers results" + '\n' + ('-' * 125))
    logging.info(time_results)
    logging.info(('-' * 125) + '\n')

    logging.info('\n' + ('-' * 125) + '\n' + "Time per Workers results LaTex" + '\n' + ('-' * 125))
    logging.info(time_results.to_latex())
    logging.info(('-' * 125) + '\n')

    # --------------------------------------------------------------------------------- #
    #                           Memory per Workers results                              #
    # --------------------------------------------------------------------------------- #

    logging.info('\n' + ('-' * 125) + '\n' + "Memory per Workers results" + '\n' + ('-' * 125))
    logging.info(memory_results)
    logging.info(('-' * 125) + '\n')

    logging.info('\n' + ('-' * 125) + '\n' + "Memory per Workers results LaTex" + '\n' + ('-' * 125))
    logging.info(memory_results.to_latex())
    logging.info(('-' * 125) + '\n')

