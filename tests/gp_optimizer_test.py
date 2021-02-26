import unittest
from src.gp_search import GPOHpSearch
from tests.objective_functions.objective_function import ObjectiveFuncHpOptimizer
import time
import numpy as np


class TestGPHpOptimizerObjFunc(unittest.TestCase):
    def test_optimize_objective_func(self):
        obj_func_hp_optimizer = ObjectiveFuncHpOptimizer()
        param_gen = GPOHpSearch(obj_func_hp_optimizer.hp_space, max_seconds=60, max_itr=1_000)

        save_kwargs = dict(
            save_name=f"obj_func_hp_opt",
            title="GPO search: Objective function",
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

        test_acc, _ = obj_func_hp_optimizer.score(obj_func_hp_optimizer.build_model(**opt_hp),
                                                  x0=opt_hp["x0"], x1=opt_hp["x1"])

        param_gen.write_optimization_to_html(show=True, **save_kwargs)

        self.assertTrue(test_acc >= 0.99, f"objective_func --> GPO result: {test_acc*100:.3f}%"
                                          f" in {elapsed_time:.2f} [s]")
        self.assertTrue(elapsed_time <= 1.1*param_gen.max_seconds)
        self.assertTrue(param_gen.current_itr <= param_gen.max_itr)


if __name__ == '__main__':
    unittest.main()
