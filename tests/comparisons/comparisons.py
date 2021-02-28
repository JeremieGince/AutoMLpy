from src.parameter_generators import SearchType
from src import logs_file_setup, log_device_setup
from tests import execute_optimisation
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

plotly_colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def compute_stats_per_dimension_table(
        max_dim: int = 10,
        iterations_per_dim: int = 10,
        **kwargs
):
    columns = ["Dimensions", *[st.name for st in SearchType]]
    iterations_results = pd.DataFrame(columns=columns)
    time_results = pd.DataFrame(columns=columns)

    for d in range(1, max_dim+1):
        logging.info(f"\n{'-'*50} {d} Dimensions {'-'*50}")
        new_iteration_results = {"Dimensions": d, **{st.name: [] for st in SearchType}}
        new_time_results = {"Dimensions": d, **{st.name: [] for st in SearchType}}

        for _search_type in SearchType:
            logging.info(f"\n{'-'*10} {_search_type.name} search {'-'*10}")
            ell_itr = []
            ell_time = []
            for itr_seed in range(iterations_per_dim):
                param_gen = execute_optimisation(
                    _search_type,
                    dim=d,
                    optimize_kwargs=dict(stop_criterion=kwargs.get("stop_criterion", 0.9)),
                    seed=itr_seed,
                )
                ell_itr.append(param_gen.current_itr)
                ell_time.append(param_gen.last_itr_elapse_time)

            new_iteration_results[_search_type.name] = (np.mean(ell_itr), np.std(ell_itr))
            new_time_results[_search_type.name] = (np.mean(ell_time), np.std(ell_time))

        iterations_results = iterations_results.append(new_iteration_results, ignore_index=True)
        time_results = time_results.append(new_time_results, ignore_index=True)

    return iterations_results, time_results


def show_stats_per_dimension(
        max_dim: int = 10,
        iterations_per_dim: int = 10,
        **kwargs
):
    iterations_results, time_results = compute_stats_per_dimension_table(max_dim, iterations_per_dim, **kwargs)
    iterations_results_mean = {
        st.name: np.array([x[0] for x in iterations_results[st.name]])
        for st in SearchType
    }
    iterations_results_std = {
        st.name: np.array([x[1] for x in iterations_results[st.name]])
        for st in SearchType
    }

    time_results_mean = {
        st.name: np.array([x[0] for x in time_results[st.name]])
        for st in SearchType
    }
    time_results_std = {
        st.name: np.array([x[1] for x in time_results[st.name]])
        for st in SearchType
    }

    # --------------------------------------------------------------------------------- #
    #                              Initialize figure                                    #
    # --------------------------------------------------------------------------------- #

    fig = go.Figure()

    for i, st in enumerate(SearchType):
        fig.add_trace(
            go.Scatter(x=iterations_results["Dimensions"],
                       y=list(iterations_results_mean[st.name]),
                       mode='lines',
                       name=f"{st.name} Search mean",
                       line_color=plotly_colors[i],)
        )
        fig.add_trace(
            go.Scatter(x=iterations_results["Dimensions"],
                       y=list(iterations_results_mean[st.name] + iterations_results_std[st.name]),
                       mode='lines',
                       fill="tonexty",
                       name=f"{st.name} Search std",
                       line_color=plotly_colors[i],)
        )
        fig.add_trace(
            go.Scatter(x=iterations_results["Dimensions"],
                       y=list(iterations_results_mean[st.name] - iterations_results_std[st.name]),
                       mode='lines',
                       fill="tonexty",
                       name=f"{st.name} Search std",
                       line_color=plotly_colors[i],)
        )

    fig.update_xaxes(title=f"Dimensions")
    fig.update_yaxes(title="Iterations [-]")

    fig.update_layout(
        # title=kwargs.get("title", ""),
        # width=1080,
        # height=750,
        autosize=True,
        margin=dict(t=150, b=150, l=150, r=150),
        template="seaborn",
    )

    # --------------------------------------------------------------------------------- #
    #                                 Add Dropdown                                      #
    # --------------------------------------------------------------------------------- #
    # fig.update_layout(
    #     updatemenus=[
    #         dict(
    #             buttons=list([
    #                 dict(
    #                     args=[
    #                         *[dict(
    #                             # x=[data_mean["Dimensions"]],
    #                             y=[list(data_mean[st.name])],
    #                         ) for st in SearchType],
    #                         {
    #                             # "title": f"{p_name}",
    #                             "xaxis.title.text": f"Dimensions [-]",
    #                             "yaxis.title.text": f"{label}",
    #                         }
    #                     ],
    #                     method="update"
    #                 )
    #                 for data_mean, label in zip([iterations_results_mean, time_results_mean],
    #                                             ["Iterations [-]", "Time [s]"])
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

    # --------------------------------------------------------------------------------- #
    #                               Saving/showing                                      #
    # --------------------------------------------------------------------------------- #
    save_dir = kwargs.get("save_dir", f"figures/html_files/")
    os.makedirs(save_dir, exist_ok=True)
    if kwargs.get("save", True):
        fig.write_html(f"{save_dir}/algorithms_comparison-maxdim{max_dim}-iteration{iterations_per_dim}.html")

    fig.show()
    return iterations_results, time_results


if __name__ == '__main__':
    import logging

    logs_file_setup(__file__, level=logging.INFO)
    log_device_setup()

    iterations_results, time_results = show_stats_per_dimension(10, 10, stop_criterion=0.9)

    # --------------------------------------------------------------------------------- #
    #                      Iteration per Dimension results                              #
    # --------------------------------------------------------------------------------- #

    logging.info('\n' + ('-' * 125) + '\n' + "Iteration per Dimension results" + '\n' + ('-' * 125))
    logging.info(iterations_results)
    logging.info(('-' * 125) + '\n')

    logging.info('\n' + ('-' * 125) + '\n' + "Iteration per Dimension results LaTex" + '\n' + ('-' * 125))
    logging.info(iterations_results.to_latex())
    logging.info(('-' * 125) + '\n')

    # --------------------------------------------------------------------------------- #
    #                             Time per Dimension results                            #
    # --------------------------------------------------------------------------------- #

    logging.info('\n' + ('-' * 125) + '\n' + "Time per Dimension results" + '\n' + ('-' * 125))
    logging.info(time_results)
    logging.info(('-' * 125) + '\n')

    logging.info('\n' + ('-' * 125) + '\n' + "Time per Dimension results LaTex" + '\n' + ('-' * 125))
    logging.info(time_results.to_latex())
    logging.info(('-' * 125) + '\n')

