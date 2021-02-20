import unittest
from modules.random_search import RandomHpSearch
from tests.objective_functions.objective_function import ObjectiveFuncHpOptimizer
from tests.pytorch_items.pytorch_datasets import get_MNIST_X_y, get_Cifar10_X_y
import time
from tests.pytorch_items.poutyne_hp_optimizers import PoutyneCifar10HpOptimizer, PoutyneMNISTHpOptimizer
import numpy as np


class TestRandomHpOptimizerObjFunc(unittest.TestCase):
    def test_optimize_objective_func(self):
        obj_func_hp_optimizer = ObjectiveFuncHpOptimizer()
        param_gen = RandomHpSearch(obj_func_hp_optimizer.hp_space, max_seconds=60, max_itr=1_000)

        start_time = time.time()
        param_gen = obj_func_hp_optimizer.optimize(
            param_gen,
            np.ones((2, 2)),
            np.ones((2, 2)),
            n_splits=2,
            save_kwargs=dict(save_name=f"obj_func_hp_opt"),
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        opt_hp = param_gen.get_best_param()

        test_acc, _ = obj_func_hp_optimizer.score(obj_func_hp_optimizer.build_model(**opt_hp),
                                                  x0=opt_hp["x0"], x1=opt_hp["x1"])

        param_gen.write_optimization_to_html(show=True, save_name="obj_func", title="Objective function")

        self.assertTrue(test_acc >= 0.9, f"objective_func --> Random Gen result: {test_acc:.2f}%"
                                         f" in {elapsed_time:.2f} [s]")
        self.assertTrue(elapsed_time <= 1.1*param_gen.max_seconds)
        self.assertTrue(param_gen.current_itr <= param_gen.max_itr)


class TestRandomHpOptimizerVisionProblem(unittest.TestCase):
    def test_optimize_Cifar10(self):
        # http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130
        cifar10_X_y_dict = get_Cifar10_X_y()

        cifar10_hp_optimizer = PoutyneCifar10HpOptimizer()

        hp_space = dict(
            epochs=[e for e in range(5, 35, 5)],
            batch_size=[32, 64],
            learning_rate=[10 ** e for e in [-3, -2, -1]],
            nesterov=[True, False],
            momentum=np.linspace(0, 0.99, 50),
            use_batchnorm=[True, False],
            pre_normalized=[True, False],
        )
        param_gen = RandomHpSearch(hp_space, max_seconds=60 * 5, max_itr=10_000)

        start_time = time.time()
        param_gen = cifar10_hp_optimizer.optimize(
            param_gen,
            cifar10_X_y_dict["train"]["x"],
            cifar10_X_y_dict["train"]["y"],
            n_splits=2,
            save_kwargs=dict(save_name=f"cifar10_hp_opt"),
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        opt_hp = param_gen.get_best_param()

        model = cifar10_hp_optimizer.build_model(**opt_hp)
        cifar10_hp_optimizer.fit_model_(model, cifar10_X_y_dict["train"]["x"], cifar10_X_y_dict["train"]["y"])

        test_acc, _ = cifar10_hp_optimizer.score(
            model,
            cifar10_X_y_dict["test"]["x"],
            cifar10_X_y_dict["test"]["y"],
            **opt_hp
        )

        param_gen.write_optimization_to_html(show=True, save_name="cifar10", title="Cifar10")

        self.assertTrue(
            test_acc >= 0.7,
            f"Cifar10 --> Random Gen result: {test_acc:.2f}%"
        )
        self.assertTrue(
            elapsed_time <= 1.15 * param_gen.max_seconds,
            f"Had a budget of {param_gen.max_seconds}s and take {elapsed_time}s"
        )
        self.assertTrue(
            param_gen.current_itr <= param_gen.max_itr,
            f"Had a budget of {param_gen.max_itr}itr and take {param_gen.current_itr}itr"
        )

    def test_optimize_MNIST(self):
        # http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130

        mnist_X_y_dict = get_MNIST_X_y()

        mnist_hp_optimizer = PoutyneMNISTHpOptimizer()

        hp_space = dict(
            epochs=[e for e in range(5, 35, 5)],
            batch_size=[32, 64],
            learning_rate=[10 ** e for e in [-3, -2, -1]],
            nesterov=[True, False],
            momentum=np.linspace(0, 0.99, 50),
            pre_normalized=[False, True],
        )
        param_gen = RandomHpSearch(hp_space, max_seconds=60*5, max_itr=10_000)

        start_time = time.time()
        param_gen = mnist_hp_optimizer.optimize(
            param_gen,
            mnist_X_y_dict["train"]["x"],
            mnist_X_y_dict["train"]["y"],
            n_splits=2,
            save_kwargs=dict(save_name=f"mnist_hp_opt"),
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        opt_hp = param_gen.get_best_param()

        model = mnist_hp_optimizer.build_model(**opt_hp)
        mnist_hp_optimizer.fit_model_(model, mnist_X_y_dict["train"]["x"], mnist_X_y_dict["train"]["y"])

        test_acc, _ = mnist_hp_optimizer.score(
            model,
            mnist_X_y_dict["test"]["x"],
            mnist_X_y_dict["test"]["y"],
            **opt_hp
        )

        param_gen.write_optimization_to_html(show=True, save_name="mnist", title="MNIST")

        self.assertTrue(
            test_acc >= 0.985,
            f"MNIST --> Random Gen result: {test_acc:.2f}%"
        )
        self.assertTrue(
            elapsed_time <= 1.15 * param_gen.max_seconds,
            f"Had a budget of {param_gen.max_seconds}s and take {elapsed_time}s"
        )
        self.assertTrue(
            param_gen.current_itr <= param_gen.max_itr,
            f"Had a budget of {param_gen.max_itr}itr and take {param_gen.current_itr}itr"
        )


if __name__ == '__main__':
    unittest.main()


