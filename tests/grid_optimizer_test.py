import unittest
from src.grid_search import GridHpSearch
from tests.objective_functions.objective_function import ObjectiveFuncHpOptimizer
from tests.pytorch_datasets import get_MNIST_X_y, get_Cifar10_X_y
import time
from tests.poutyne_hp_optimizers import PoutyneCifar10HpOptimizer, PoutyneMNISTHpOptimizer
import numpy as np


class TestGridHpOptimizer(unittest.TestCase):
    def test_optimize_result_objective_func(self):
        obj_func_hp_optimizer = ObjectiveFuncHpOptimizer()
        param_gen = GridHpSearch(obj_func_hp_optimizer.hp_space, max_seconds=60)

        start_time = time.time()
        param_gen = obj_func_hp_optimizer.optimize(param_gen, np.ones((2, 2)), np.ones((2, 2)), n_splits=2)
        end_time = time.time()
        elapsed_time = end_time - start_time

        opt_hp = param_gen.get_best_param()

        test_acc, _ = obj_func_hp_optimizer.score(obj_func_hp_optimizer.build_model(), x0=opt_hp["x0"], x1=opt_hp["x1"])

        self.assertTrue(test_acc >= 0.9, f"objective_func --> Grid Gen result: {test_acc:.2f}%"
                                         f" in {elapsed_time:.2f} [s]")
        self.assertTrue(elapsed_time <= 1.15 * param_gen.max_seconds)
        self.assertTrue(param_gen.current_itr <= param_gen.max_itr)

    def test_optimize_Cifar10(self):
        # http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130
        cifar10_X_y_dict = get_Cifar10_X_y()

        cifar10_hp_optimizer = PoutyneCifar10HpOptimizer()

        hp_space = dict(
            epochs=[e for e in range(5, 105, 5)],
            batch_size=[32, 64],
            learning_rate=[10 ** e for e in [-3, -2, -1]],
        )
        param_gen = GridHpSearch(hp_space, max_seconds=90, max_itr=100)

        start_time = time.time()
        param_gen = cifar10_hp_optimizer.optimize(
            param_gen,
            cifar10_X_y_dict["train"]["x"],
            cifar10_X_y_dict["train"]["y"],
            n_splits=2
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        opt_hp = param_gen.get_best_param()

        test_acc, _ = cifar10_hp_optimizer.score(
            cifar10_hp_optimizer.build_model(**opt_hp),
            cifar10_X_y_dict["test"]["x"],
            cifar10_X_y_dict["test"]["y"],
            **opt_hp
        )

        self.assertTrue(
            test_acc >= 0.9,
            f"Cifar10 --> Random Gen result: {test_acc:.2f}% in {elapsed_time:.2f} [s]"
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
            epochs=[e for e in range(5, 105, 5)],
            batch_size=[32, 64],
            learning_rate=[10 ** e for e in [-3, -2, -1]],
        )
        param_gen = GridHpSearch(hp_space, max_seconds=90, max_itr=100)

        start_time = time.time()
        param_gen = mnist_hp_optimizer.optimize(
            param_gen,
            mnist_X_y_dict["train"]["x"],
            mnist_X_y_dict["train"]["y"],
            n_splits=2
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        opt_hp = param_gen.get_best_param()

        test_acc, _ = mnist_hp_optimizer.score(
            mnist_hp_optimizer.build_model(**opt_hp),
            mnist_X_y_dict["test"]["x"],
            mnist_X_y_dict["test"]["y"],
            **opt_hp
        )

        self.assertTrue(
            test_acc >= 0.9,
            f"MNIST --> Random Gen result: {test_acc:.2f}% in {elapsed_time:.2f} [s]"
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
