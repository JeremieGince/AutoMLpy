import unittest
from src.AutoMLpy import RandomHpSearch

# Pytorch
from tests.objective_functions.objective_function import ObjectiveFuncHpOptimizer
from tests.objective_functions.vectorized_objective_function import VectorizedObjectiveFuncHpOptimizer
from tests.pytorch_items.pytorch_datasets import get_torch_MNIST_X_y, get_torch_Cifar10_X_y
from tests.pytorch_items.pytorch_hp_optimizers import TorchCifar10HpOptimizer, TorchMNISTHpOptimizer

# Tensorflow
from tests.tensorflow_items.tf_datasets import get_tf_mnist_dataset
from tests.tensorflow_items.tf_hp_optimizers import KerasMNISTHpOptimizer

# Utility modules
import time
import numpy as np


class TestRandomHpOptimizerObjFunc(unittest.TestCase):
    def test_optimize_objective_func(self):
        obj_func_hp_optimizer = ObjectiveFuncHpOptimizer()
        param_gen = RandomHpSearch(obj_func_hp_optimizer.hp_space, max_seconds=60, max_itr=1_000)

        save_kwargs = dict(
            save_name=f"obj_func_hp_opt",
            title="Random search: Objective function",
        )

        start_time = time.time()
        param_gen = obj_func_hp_optimizer.optimize(
            param_gen,
            np.ones((2, 2)),
            np.ones((2, 2)),
            n_splits=2,
            save_kwargs=save_kwargs,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        opt_hp = param_gen.get_best_param()

        test_acc = obj_func_hp_optimizer.score(obj_func_hp_optimizer.build_model(**opt_hp),
                                               x0=opt_hp["x0"], x1=opt_hp["x1"])

        param_gen.write_optimization_to_html(show=True, **save_kwargs)

        self.assertTrue(test_acc >= 0.99, f"objective_func --> Random Gen result: {test_acc*100:.3f}%"
                                          f" in {elapsed_time:.2f} [s]")
        self.assertTrue(elapsed_time <= 1.1*param_gen.max_seconds)
        self.assertTrue(param_gen.current_itr <= param_gen.max_itr)

    def test_vectorized_optimize_objective_func(self):
        np.random.seed(42)

        obj_func_hp_optimizer = VectorizedObjectiveFuncHpOptimizer()
        obj_func_hp_optimizer.set_dim(3)
        _ = obj_func_hp_optimizer.build_model()
        param_gen = RandomHpSearch(obj_func_hp_optimizer.hp_space, max_seconds=60*15, max_itr=10_000)

        save_kwargs = dict(
            save_name=f"vec_obj_func_hp_opt",
            title="Random search: Vectorized objective function",
        )

        start_time = time.time()
        param_gen = obj_func_hp_optimizer.optimize(
            param_gen,
            np.ones((2, 2)),
            np.ones((2, 2)),
            n_splits=2,
            save_kwargs=save_kwargs,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        opt_hp = param_gen.get_best_param()

        test_acc = obj_func_hp_optimizer.score(obj_func_hp_optimizer.build_model(**opt_hp), **opt_hp)

        param_gen.write_optimization_to_html(show=True, **save_kwargs)

        self.assertTrue(test_acc >= 0.99, f"Vectorized objective_func --> Random Gen result: {test_acc*100:.3f}%"
                                          f" in {elapsed_time:.2f} [s]")
        self.assertTrue(elapsed_time <= 1.1*param_gen.max_seconds)
        self.assertTrue(param_gen.current_itr <= param_gen.max_itr)


class TestRandomHpOptimizerVisionProblemPytorch(unittest.TestCase):
    def test_optimize_Cifar10(self):
        # http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130
        cifar10_X_y_dict = get_torch_Cifar10_X_y()

        cifar10_hp_optimizer = TorchCifar10HpOptimizer()

        hp_space = dict(
            epochs=list(range(1, 26)),
            batch_size=[32, 64],
            learning_rate=np.linspace(1e-4, 1e-1, 50),
            nesterov=[True, False],
            momentum=np.linspace(0.01, 0.99, 50),
            use_batchnorm=[True, False],
            pre_normalized=[True, False],
        )
        param_gen = RandomHpSearch(hp_space, max_seconds=60 * 60 * 1, max_itr=1_000)

        save_kwargs = dict(
            save_name=f"cifar10_hp_opt",
            title="Random search: Cifar10",
        )

        start_time = time.time()
        param_gen = cifar10_hp_optimizer.optimize(
            param_gen,
            cifar10_X_y_dict["train"]["x"],
            cifar10_X_y_dict["train"]["y"],
            n_splits=2,
            stop_criterion=0.75,
            save_kwargs=save_kwargs,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        opt_hp = param_gen.get_best_param()

        model = cifar10_hp_optimizer.build_model(**opt_hp)
        cifar10_hp_optimizer.fit_model_(
            model,
            cifar10_X_y_dict["train"]["x"],
            cifar10_X_y_dict["train"]["y"],
            **opt_hp
        )

        test_acc = cifar10_hp_optimizer.score(
            model.cpu(),
            cifar10_X_y_dict["test"]["x"],
            cifar10_X_y_dict["test"]["y"],
            **opt_hp
        )

        param_gen.write_optimization_to_html(show=True, **save_kwargs)

        self.assertTrue(
            test_acc >= 0.7,
            f"Cifar10 --> Random Gen result: {test_acc*100:.3f}%"
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

        mnist_X_y_dict = get_torch_MNIST_X_y()
        mnist_hp_optimizer = TorchMNISTHpOptimizer()

        hp_space = dict(
            epochs=list(range(1, 16)),
            batch_size=[32, 64, 128],
            learning_rate=np.linspace(1e-4, 1e-1, 50),
            nesterov=[True, False],
            momentum=np.linspace(0.01, 0.99, 50),
            pre_normalized=[False, True],
        )
        param_gen = RandomHpSearch(hp_space, max_seconds=60*60*1, max_itr=1_000)

        save_kwargs = dict(
            save_name=f"mnist_hp_opt",
            title="Random search: MNIST",
        )

        start_time = time.time()
        param_gen = mnist_hp_optimizer.optimize(
            param_gen,
            mnist_X_y_dict["train"]["x"],
            mnist_X_y_dict["train"]["y"],
            n_splits=2,
            stop_criterion=0.99,
            save_kwargs=save_kwargs,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        opt_hp = param_gen.get_best_param()

        model = mnist_hp_optimizer.build_model(**opt_hp)
        mnist_hp_optimizer.fit_model_(
            model,
            mnist_X_y_dict["train"]["x"],
            mnist_X_y_dict["train"]["y"],
            **opt_hp
        )

        test_acc = mnist_hp_optimizer.score(
            model.cpu(),
            mnist_X_y_dict["test"]["x"],
            mnist_X_y_dict["test"]["y"],
            **opt_hp
        )

        param_gen.write_optimization_to_html(show=True, **save_kwargs)

        self.assertTrue(
            test_acc >= 0.985,
            f"MNIST --> Random Gen result: {test_acc*100:.3f}%"
        )
        self.assertTrue(
            elapsed_time <= 1.15 * param_gen.max_seconds,
            f"Had a budget of {param_gen.max_seconds}s and take {elapsed_time}s"
        )
        self.assertTrue(
            param_gen.current_itr <= param_gen.max_itr,
            f"Had a budget of {param_gen.max_itr}itr and take {param_gen.current_itr}itr"
        )


class TestRandomHpOptimizerVisionProblemTensorflow(unittest.TestCase):
    def test_optimize_MNIST(self):
        # http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130

        mnist_train, mnist_test = get_tf_mnist_dataset()

        mnist_hp_optimizer = KerasMNISTHpOptimizer()

        hp_space = dict(
            epochs=list(range(1, 16)),
            learning_rate=np.linspace(1e-4, 1e-1, 50),
            nesterov=[True, False],
            momentum=np.linspace(0.01, 0.99, 50),
            use_conv=[True, False],
        )
        param_gen = RandomHpSearch(hp_space, max_seconds=60*60*1, max_itr=1_000)

        save_kwargs = dict(
            save_name=f"tf_mnist_hp_opt",
            title="Random search: MNIST",
        )

        start_time = time.time()
        param_gen = mnist_hp_optimizer.optimize_on_dataset(
            param_gen, mnist_train, save_kwargs=save_kwargs, stop_criterion=0.99,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        opt_hp = param_gen.get_best_param()

        model = mnist_hp_optimizer.build_model(**opt_hp)
        mnist_hp_optimizer.fit_dataset_model_(
            model, mnist_train, **opt_hp
        )

        test_acc = mnist_hp_optimizer.score_on_dataset(
            model, mnist_test, **opt_hp
        )

        param_gen.write_optimization_to_html(show=True, **save_kwargs)

        self.assertTrue(
            test_acc >= 0.985,
            f"MNIST --> Random Gen result: {test_acc*100:.3f}%"
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


