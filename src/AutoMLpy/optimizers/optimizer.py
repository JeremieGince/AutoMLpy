import warnings
import multiprocessing
import tqdm
from sklearn.model_selection import KFold
from typing import Union, List, Tuple
from multiprocessing import Pool
import logging
import numpy as np
import time

optional_modules = {}
try:
    import pandas as pd
    optional_modules["pandas"] = True
except ModuleNotFoundError:
    optional_modules["pandas"] = False

try:
    import torch
    optional_modules["torch"] = True
except ModuleNotFoundError:
    optional_modules["torch"] = False

try:
    import tensorflow as tf
    optional_modules["tensorflow"] = True
except ModuleNotFoundError:
    optional_modules["tensorflow"] = False

from ..parameter_generators import ParameterGenerator


class ParamGenType:
    Grid = 0
    Random = 1
    GPO = 2


class HpOptimizer:
    """
    Class used to optimize a set of hyper-parameters with a parameter generator. This is a virtual class
    and the user must implement the following method before the optimization:

        To work with data in form X and y:
            build_model: method used to build the model to optimize given a set of hyper-parameters,
            fit_model_: method used to fit the model to optimize given a set of hyper-parameters,
            score: method used to get the score of the trained model given a set of hyper-parameters,

        To work with datasets:
            build_model: method used to build the model to optimize given a set of hyper-parameters,
            fit_dataset_model_: method used to fit the model to optimize given a set of hyper-parameters,
            score_on_dataset: method used to get the score of the trained model given a set of hyper-parameters,

    Examples on how to use this class are in the folder "./examples".

    """
    def build_model(self, **hp) -> object:
        """
        Method used to build the model to optimize given a set of hyper-parameters.

        Parameters
        ----------
        hp: The hyper-parameters that come from the parameter generator.

        Returns
        -------
        The model to optimized.
        """
        raise NotImplementedError("build_model method must be implemented by the user")

    def fit_model_(
            self,
            model: object,
            X, y,
            X_val=None, y_val=None,
            **hp
    ) -> object:
        """
        Method used to fit the model to optimize given a set of hyper-parameters.

        Parameters
        ----------
        model: The model to fitted. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        X: The training input data. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        y: The training labels. (Union[np.ndarray, torch.Tensor, tf.Tensor])
        X_val: The validation input data. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        y_val: The validation labels. (Union[np.ndarray, torch.Tensor, tf.Tensor])
        hp: The hyper-parameters that come from the parameter generator.

        Returns
        -------
        The fitted model.
        """
        raise NotImplementedError("fit_model_ method must be implemented by the user")

    def score(
            self,
            model: object,
            X, y,
            **hp
    ) -> float:
        """
        Method used to get the score of the trained model given a set of hyper-parameters.

        Parameters
        ----------
        model: The model to fitted. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        X: The training input data. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        y: The training labels. (Union[np.ndarray, torch.Tensor, tf.Tensor])
        hp: The hyper-parameters that come from the parameter generator.

        Returns
        -------
        The normalized score as a float between 0 and 1.
        """
        raise NotImplementedError("score method must be implemented by the user")

    def fit_dataset_model_(
            self,
            model: object,
            dataset,
            **hp
    ) -> object:
        """
        Method used to fit the model to optimize given a set of hyper-parameters.

        Parameters
        ----------
        model: The model to fitted. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        dataset: The dataset used to train the model. (Union[torch.utils.data.Dataset, tf.data.Dataset])
        hp: The hyper-parameters that come from the parameter generator.

        Returns
        -------
        The fitted model.
        """
        raise NotImplementedError("fit_dataset_model_ method must be implemented by the user")

    def score_on_dataset(
            self,
            model: object,
            dataset,
            **hp
    ) -> float:
        """
        Method used to get the score of the trained model given a set of hyper-parameters.

        Parameters
        ----------
        model: The model to fitted. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        dataset: The dataset used to train the model. (Union[torch.utils.data.Dataset, tf.data.Dataset])
        hp: The hyper-parameters that come from the parameter generator.

        Returns
        -------
        The normalized score as a float between 0 and 1.
        """
        raise NotImplementedError("score_dataset method must be implemented by the user")

    def __call__(self, *args, **kwargs):
        return self.optimize(*args, **kwargs)

    def optimize(
            self,
            param_gen: ParameterGenerator,
            X, y,
            n_splits: int = 2,
            nb_workers: Union[int, str] = 1,
            save_kwargs: dict = None,
            verbose: bool = True,
            **kwargs
    ) -> ParameterGenerator:
        """
        Optimize hyper-parameters using the given ParameterGenerator with kfolding.

        Parameters
        ----------
        param_gen: The parameter generator.
        X: The training data. (Union[np.ndarray, torch.Tensor, pd.Dataframe, tf.Tensor])
        y: The training labels. (Union[np.ndarray, torch.Tensor])
        n_splits: Number of split for the kfold. The X_val and the y_val data will be generated by the kfold and
                  pass to the fit method ans the score method. If n_splits is set ot <= 1, no kfold is done and the
                  inputs X, y will be pass to the fit method ans the score method.
        nb_workers: number of used cpu. Must be a int or the string "max". Default: 1.
        save_kwargs: Saving kwargs of the parameter generator.
        verbose: True to print some stats else False.
        kwargs:
            stop_criterion: If the score once reach this criterion the optimization will stop. (float)
            reset_gen: True to reset the parameter generator before the optimization else False. (bool) Default: False.
            minimise: True to minimise the score and False to maximise it. If specified, will update the value of
                        param_gen.minimise. Default: take the value of param_gen.minimise.

        Return
        ---------
        The given ParameterGenerator optimized.
        """
        if save_kwargs is None:
            save_kwargs = {}
        save_kwargs.setdefault("save_each_run", 1)

        if kwargs.get("filterwarnings", True):
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)

        param_gen.minimise = kwargs.get("minimise", param_gen.minimise)
        nb_workers = self._setup_nb_workers(nb_workers, verbose, **kwargs)

        stop_criterion = kwargs.get("stop_criterion", None)
        stop_criterion_trigger = False

        if kwargs.get("reset_gen", False):
            param_gen.reset()

        progress_bar = tqdm.tqdm(range(param_gen.max_itr), unit='itr', postfix="optimisation") if verbose else None
        progress_bar.update(param_gen.current_itr)
        while bool(param_gen) and not stop_criterion_trigger:
            try:
                workers_required = min(nb_workers, param_gen.max_itr - param_gen.current_itr)
                outputs = self._process_trial_on_X_y(param_gen, X, y, n_splits, workers_required)

                stop_criterion_trigger = self._post_process_trial_(
                    param_gen, outputs, stop_criterion, progress_bar, verbose
                )
                if param_gen.current_itr % save_kwargs["save_each_run"] == 0:
                    param_gen.save_obj(**save_kwargs)

            except Exception as ee:
                logging.error(str(ee))
                raise ee

        if verbose:
            progress_bar.close()

        param_gen.save_obj(**save_kwargs)
        param_gen.save_history(**save_kwargs)
        param_gen.save_best_param(**save_kwargs)
        param_gen.write_optimization_to_html(show=False, **save_kwargs)
        return param_gen

    def _process_trial_on_X_y(
            self,
            param_gen: ParameterGenerator,
            X, y,
            n_splits: int,
            workers_required: int,
    ) -> List[Tuple[dict, float]]:
        """
        Execute a trial on a set of parameters and a X, y for data.

        Parameters
        ----------
        param_gen: The parameter generator.
        X: The training input data. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        y: The training labels. (Union[np.ndarray, torch.Tensor, tf.Tensor])
        n_splits: Number of split for the kfold.
        workers_required: Number of used cpu. Must be a int or the string "max". Default: 1.

        Returns
        -------
        The list of outputs made by the multiple trials as list of tuples (parameters, score).
        """
        if workers_required > 1:
            outputs = self._execute_multiple_param_gen_iteration_on_X_y(workers_required, param_gen, X, y, n_splits)
        else:
            outputs = [self._execute_param_gen_iteration_on_X_y(param_gen, X, y, n_splits)]
        return outputs

    def _post_process_trial_(
            self,
            param_gen: ParameterGenerator,
            trial_outputs: List[Tuple[dict, float]],
            stop_criterion: float,
            progress_bar: tqdm.tqdm,
            verbose: bool,
    ) -> bool:
        """
        Post process the outputs of the trial execution of the training.
        Update param_gen, logs, progress and stop criterion trigger state.

        Parameters
        ----------
        param_gen: The parameter generator.
        trial_outputs: The list of outputs made by the multiple trials as list of tuples (parameters, score).
        stop_criterion: If the score once reach this criterion the optimization will stop. (float)
        progress_bar: Progress bar.
        verbose: True to print some stats else False.

        Returns
        -------
        The stop_criterion_trigger.
        """
        stop_criterion_trigger = False
        if param_gen.minimise:
            check_stop_criterion = lambda v: stop_criterion >= v
        else:
            check_stop_criterion = lambda v: stop_criterion <= v

        for (params, mean_score) in trial_outputs:
            param_gen.add_score_info(params, mean_score)

            if verbose:
                logging.debug(f"param_gen: curr_itr: {param_gen.current_itr}")
                logging.debug(f"trial_params: {params} --> mean score: {mean_score:.3f}")
                progress_bar.set_postfix_str(f"mean_score: {mean_score:.2f}")
                progress_bar.update()

            if stop_criterion is not None and check_stop_criterion(mean_score):
                if verbose:
                    logging.info(f"Early stopping ->"
                                 f" itr: {param_gen.current_itr},"
                                 f" elapse_time: {param_gen.elapse_time:.3f} [s],"
                                 f" mean score: {mean_score:.3f}")
                stop_criterion_trigger = True
        return stop_criterion_trigger

    def optimize_on_dataset(
            self,
            param_gen: ParameterGenerator,
            dataset,
            nb_workers: Union[int, str] = 1,
            save_kwargs: dict = None,
            verbose: bool = True,
            **kwargs
    ) -> ParameterGenerator:
        """
        Optimize hyper-paremeters using the given ParameterGenerator and kfolding.

        Parameters
        ----------
        param_gen: The parameter generator.
        dataset: The dataset used to train the model. (Union[torch.utils.data.Dataset, tf.data.Dataset])
        nb_workers: Number of used cpu. Must be a int or the string "max". Default: 1.
        save_kwargs: Saving kwargs of the parameter generator.
        verbose: True to print some stats else False.
        kwargs:
            stop_criterion: If the score once reach this criterion the optimization will stop. (float)
            reset_gen: True to reset the parameter generator before the optimization else False. (bool) Default: False.
            minimise: True to minimise the score and False to maximise it. If specified, will update the value of
                        param_gen.minimise. Default: take the value of param_gen.minimise.

        Return
        ---------
        The given ParameterGenerator optimized.
        """
        if save_kwargs is None:
            save_kwargs = {}
        save_kwargs.setdefault("save_each_run", 1)
        warnings.simplefilter("ignore", UserWarning)

        param_gen.minimise = kwargs.get("minimise", param_gen.minimise)
        nb_workers = self._setup_nb_workers(nb_workers, verbose, **kwargs)

        stop_criterion = kwargs.get("stop_criterion", None)
        stop_criterion_trigger = False

        if kwargs.get("reset_gen", False):
            param_gen.reset()

        progress_bar = tqdm.tqdm(range(param_gen.max_itr), unit='itr', postfix="optimisation") if verbose else None
        progress_bar.update(param_gen.current_itr)
        while bool(param_gen) and not stop_criterion_trigger:
            try:
                workers_required = min(nb_workers, param_gen.max_itr - param_gen.current_itr)
                outputs = self._process_trial_on_dataset(param_gen, dataset, workers_required)

                stop_criterion_trigger = self._post_process_trial_(
                    param_gen, outputs, stop_criterion, progress_bar, verbose
                )
                if param_gen.current_itr % save_kwargs["save_each_run"] == 0:
                    param_gen.save_obj(**save_kwargs)

            except Exception as ee:
                logging.error(str(ee))
                raise ee

        if verbose:
            progress_bar.close()

        param_gen.save_obj(**save_kwargs)
        param_gen.save_history(**save_kwargs)
        param_gen.save_best_param(**save_kwargs)
        param_gen.write_optimization_to_html(show=False, **save_kwargs)
        return param_gen

    def _process_trial_on_dataset(
            self,
            param_gen: ParameterGenerator,
            dataset,
            workers_required: int,
    ) -> List[Tuple[dict, float]]:
        """
        Execute a trial on a set of parameters and a dataset for data.

        Parameters
        ----------
        param_gen: The parameter generator.
        dataset: The dataset used to train the model.
        workers_required: Number of used cpu. Must be a int or the string "max". Default: 1.

        Returns
        -------
        The list of outputs made by the multiple trials as list of tuples (parameters, score).
        """
        if workers_required > 1:
            outputs = self._execute_multiple_param_gen_iteration_on_dataset(workers_required, param_gen, dataset)
        else:
            outputs = [self._execute_param_gen_iteration_on_dataset(param_gen, dataset)]
        return outputs

    @staticmethod
    def _setup_nb_workers(
            nb_workers: Union[int, str] = 1,
            verbose: bool = True,
            **kwargs
    ):
        """
        Check how many worker is available given the ideal number of workers.

        Parameters
        ----------
        nb_workers: Ideal number of workers.
        verbose: True to show logs else False.
        kwargs:
            None

        Returns
        -------
        The number of available workers or the numbers of ideal workers.
        """
        if isinstance(nb_workers, str):
            if nb_workers.lower() == "max":
                nb_workers = multiprocessing.cpu_count()
        else:
            nb_workers = min(multiprocessing.cpu_count(), nb_workers)

        if verbose:
            logging.info(f"Number of available cpu : {multiprocessing.cpu_count()} --> Using {nb_workers} cpu")
        return nb_workers

    def _execute_multiple_param_gen_iteration_on_X_y(
            self,
            nb_workers: int,
            param_gen: ParameterGenerator,
            X, y,
            n_splits: int = 2,
    ) -> List[Tuple[dict, float]]:
        """
        Execute multiple trials on with the parameter generator on the data X, y.
        The trials are made in multiprocessing with the number of workers.

        Parameters
        ----------
        nb_workers: Number of multiprocessing process to start.
        param_gen: The current parameter generator.
        X: The training input data. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        y: The training labels. (Union[np.ndarray, torch.Tensor, tf.Tensor])
        n_splits: Number of split for the kfold.

        Returns
        -------
        The list of outputs made by the multiple trials as list of tuples (parameters, score).
        """
        params_list = [param_gen.get_trial_param(worker_id=i) for i in range(nb_workers)]
        with Pool(nb_workers) as p:
            scores = p.starmap(self._try_params_on_X_y, [
                (params, X, y, n_splits)
                for params in params_list
            ])
        outputs = [(params, score) for params, score in zip(params_list, scores)]
        return outputs

    def _execute_param_gen_iteration_on_X_y(
            self,
            param_gen: ParameterGenerator,
            X, y,
            n_splits: int = 2,
    ) -> Tuple[dict, float]:
        """
        Execute a trial on with the parameter generator on the data X, y.

        Parameters
        ----------
        param_gen: The current parameter generator.
        X: The training input data. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        y: The training labels. (Union[np.ndarray, torch.Tensor, tf.Tensor])
        n_splits: Number of split for the kfold.

        Returns
        -------
        The output of the trial as a tuple (parameters, score).
        """
        params = param_gen.get_trial_param()
        mean_score = self._try_params_on_X_y(params, X, y, n_splits)
        return params, mean_score

    def _try_params_on_X_y(
            self,
            params: dict,
            X, y,
            n_splits: int = 2,
    ) -> float:
        """
        Try a set of hyper-parameters on the model by using the methods implemented by the user.

        Parameters
        ----------
        params: The trial hyper-parameters.
        X: The training input data. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        y: The training labels. (Union[np.ndarray, torch.Tensor, tf.Tensor])
        n_splits: Number of split for the kfolding.

        Returns
        -------
        The output score of the trial.
        """
        if n_splits <= 1:
            return self._try_params_on_X_y_without_kfold(params, X, y)

        kf = KFold(n_splits=n_splits, shuffle=True)

        mean_score = 0.0
        for j, (train_index, test_index) in enumerate(kf.split(X)):
            try:
                (sub_X_train, sub_X_test), (sub_y_train, sub_y_test) = self._take_sub_X_y(X, y, train_index, test_index)

                model = self.build_model(**params)
                self.fit_model_(model, sub_X_train, sub_y_train, **params)

                score = self.score(model, sub_X_test, sub_y_test, **params)
                mean_score = (j * mean_score + score) / (j + 1)
            except Exception as e:
                logging.error(str(e))
                raise e

        return mean_score

    def _try_params_on_X_y_without_kfold(
            self,
            params: dict,
            X, y,
    ) -> float:
        """
        Try a set of hyper-parameters on the model by using the methods implemented by the user.

        Parameters
        ----------
        params: The trial hyper-parameters.
        X: The training input data. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        y: The training labels. (Union[np.ndarray, torch.Tensor, tf.Tensor])

        Returns
        -------
        The output score of the trial.
        """
        try:
            model = self.build_model(**params)
            self.fit_model_(model, X, y, **params)
            score = self.score(model, X, y, **params)
        except Exception as e:
            logging.error(str(e))
            raise e
        return score

    def _take_sub_X_y(
            self,
            X, y,
            train_index,
            test_index,
    ) -> Tuple[tuple, tuple]:
        """
        Take a subset of the entries X and y.

        Parameters
        ----------
        X: The training input data. (Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor])
        y: The training labels. (Union[np.ndarray, torch.Tensor, tf.Tensor])
        train_index: The train indexes.
        test_index: The test indexes.

        Returns
        -------
        The subsets as (sub_X_train, sub_X_test), (sub_y_train, sub_y_test).
        """
        if isinstance(X, np.ndarray):
            sub_X_train, sub_X_test = X[train_index], X[test_index]
            sub_y_train, sub_y_test = y[train_index], y[test_index]
        elif optional_modules["torch"] and isinstance(X, torch.Tensor):
            sub_X_train, sub_X_test = X[train_index], X[test_index]
            sub_y_train, sub_y_test = y[train_index], y[test_index]
        elif optional_modules["tensorflow"] and isinstance(X, tf.Tensor):
            sub_X_train, sub_X_test = X[train_index], X[test_index]
            sub_y_train, sub_y_test = y[train_index], y[test_index]
        elif optional_modules["pandas"] and isinstance(X, pd.DataFrame):
            sub_X_train, sub_X_test = X.iloc[train_index], X.iloc[test_index]
            sub_y_train, sub_y_test = y[train_index], y[test_index]
        else:
            raise ValueError(f"X must be Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]")
        return (sub_X_train, sub_X_test), (sub_y_train, sub_y_test)

    def _execute_multiple_param_gen_iteration_on_dataset(
            self,
            nb_workers: int,
            param_gen: ParameterGenerator,
            dataset,
    ) -> List[Tuple[dict, float]]:
        """
        Execute multiple trials on with the parameter generator on the data X, y.
        The trials are made in multiprocessing with the number of workers.

        Parameters
        ----------
        nb_workers: Number of multiprocessing process to start.
        param_gen: The current parameter generator.
        dataset: The dataset used to train the model.

        Returns
        -------
        The list of outputs made by the multiple trials as list of tuples (parameters, score).
        """
        params_list = [param_gen.get_trial_param(i) for i in range(nb_workers)]
        with Pool(nb_workers) as p:
            scores = p.starmap(self._try_params_on_dataset, [
                (params, dataset)
                for params in params_list
            ])
        outputs = [(params, score) for params, score in zip(params_list, scores)]
        return outputs

    def _execute_param_gen_iteration_on_dataset(
            self,
            param_gen: ParameterGenerator,
            dataset,
    ) -> Tuple[dict, float]:
        """
        Execute a trial on with the parameter generator on the dataset.

        Parameters
        ----------
        param_gen: The current parameter generator.
        dataset: The dataset used to train the model. (Union[torch.utils.data.Dataset, tf.data.Dataset])

        Returns
        -------
        The output of the trial as a tuple (parameters, score).
        """
        params = param_gen.get_trial_param()
        mean_score = self._try_params_on_dataset(params, dataset)
        return params, mean_score

    def _try_params_on_dataset(
            self,
            params: dict,
            dataset,
    ) -> float:
        """
        Try a set of hyper-parameters on the model by using the methods implemented by the user.

        Parameters
        ----------
        params The trial hyper-parameters.
        dataset: The dataset used to train the model. (Union[torch.utils.data.Dataset, tf.data.Dataset])

        Returns
        -------
        The output score of the trial.
        """
        n_splits = 2
        train_size = int(len(dataset)//n_splits)
        test_size = len(dataset) - train_size

        mean_score = 0.0
        for k in range(n_splits):
            try:
                sub_dataset_train, sub_dataset_test = self._take_sub_dataset(dataset, train_size, test_size, k)

                model = self.build_model(**params)
                self.fit_dataset_model_(model, sub_dataset_train, **params)

                score = self.score_on_dataset(model, sub_dataset_test, **params)
                mean_score = (k * mean_score + score) / (k + 1)
            except Exception as e:
                logging.error(str(e))
                raise e

        return mean_score

    def _take_sub_dataset(
            self,
            dataset,
            train_size: int,
            test_size: int,
            k: int
    ) -> Tuple:
        """
        Take subsets of the input dataset.

        Parameters
        ----------
        dataset: The dataset used to train the model. (Union[torch.utils.data.Dataset, tf.data.Dataset])
        train_size: Size of the train subset.
        test_size: Size of the test subset.
        k: The shift index.

        Returns
        -------
        The subsets as (sub_dataset_train, sub_dataset_test).
        """
        if optional_modules["torch"] and isinstance(dataset, torch.Tensor):
            raise NotImplementedError("torch dataset is not implemented yet")
        elif optional_modules["tensorflow"] and isinstance(dataset, tf.data.Dataset):
            dataset = dataset.shuffle(len(dataset))
            if k == 0:
                sub_dataset_train = dataset.take(train_size)
                sub_dataset_test = dataset.skip(train_size)
            elif k == 1:
                sub_dataset_test = dataset.take(test_size)
                sub_dataset_train = dataset.skip(test_size)
            else:
                raise ValueError(f"k must be equal to 0 or 1")
        else:
            raise ValueError(f"dataset is {type(dataset)} but must be Union[tf.data.Dataset, ]")
        return sub_dataset_train, sub_dataset_test


if __name__ == '__main__':
    from src.AutoMLpy import RandomHpSearch
    from tests.pytorch_items.pytorch_datasets import get_torch_Cifar10_X_y
    from tests.pytorch_items.pytorch_hp_optimizers import TorchCifar10HpOptimizer
    from src.AutoMLpy.logging_tools import logs_file_setup, log_device_setup, DeepLib

    logs_file_setup(__file__)
    log_device_setup(DeepLib.Pytorch)

    cifar10_X_y_dict = get_torch_Cifar10_X_y()
    cifar10_hp_optimizer = TorchCifar10HpOptimizer()

    hp_space = dict(
        epochs=list(range(1, 26)),
        batch_size=[32, 64],
        learning_rate=[10 ** e for e in [-3, -2, -1]],
        nesterov=[True, False],
        momentum=np.linspace(0.01, 0.99, 50),
        # use_batchnorm=[True, False],
        pre_normalized=[True, False],
    )
    _param_gen = RandomHpSearch(hp_space, max_seconds=60 * 60 * 0.5, max_itr=10)

    start_time = time.time()
    _param_gen = cifar10_hp_optimizer.optimize(
        _param_gen,
        cifar10_X_y_dict["train"]["x"],
        cifar10_X_y_dict["train"]["y"],
        n_splits=2,
        save_kwargs=dict(save_name=f"cifar10_hp_opt"),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    opt_hp = _param_gen.get_best_param()

    logging.info(f"Predicted best hyper-parameters: \n{_param_gen.get_best_params_repr()}")
    model_ = cifar10_hp_optimizer.build_model(**opt_hp)
    cifar10_hp_optimizer.fit_model_(
        model_,
        cifar10_X_y_dict["train"]["x"],
        cifar10_X_y_dict["train"]["y"],
        verbose=True,
        **opt_hp
    )

    test_acc, _ = cifar10_hp_optimizer.score(
        model_,
        cifar10_X_y_dict["test"]["x"],
        cifar10_X_y_dict["test"]["y"],
        **opt_hp
    )

    _param_gen.write_optimization_to_html(show=True, save_name="cifar10", title="Cifar10")

    logging.info(f"test accuracy: {test_acc*100:.3f}%")
