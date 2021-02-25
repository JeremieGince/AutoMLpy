import unittest
from src.random_search import RandomHpSearch
from tests.objective_functions.objective_function import ObjectiveFuncHpOptimizer
import time
import numpy as np


class TestHpOptimizer(unittest.TestCase):
    def test_optimize_random_gen_result_objective_func(self):
        start_time = time.time()
        obj_func_hp_optimizer = ObjectiveFuncHpOptimizer()
        param_gen = RandomHpSearch(obj_func_hp_optimizer.hp_space, max_seconds=60)

        param_gen = obj_func_hp_optimizer.optimize(param_gen, np.ones((2, 2)), np.ones((2, 2)), n_splits=2)
        opt_hp = param_gen.get_best_param()

        test_acc, _ = obj_func_hp_optimizer.score(obj_func_hp_optimizer.build_model(), x0=opt_hp["x0"], x1=opt_hp["x1"])

        self.assertTrue(test_acc >= 0.9, f"Random Gen result: {test_acc:.2f}%"
                                         f" in {time.time() - start_time:.2f} [s]")
        print(f"{opt_hp}, test_acc: {test_acc}")


    # def test_optimize_random_gen_result(self):
    #     # http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130
    #     start_time = time.time()
    #
    #     hp_space = dict(
    #         epoch=[e for e in range(5, 105, 5)],
    #         batch_size=[32, 64],
    #         learning_rate=[10**e for e in [-3, -2, -1]],
    #     )
    #     param_gen = RandomHpSearch(hp_space, max_seconds=60**2)
    #     cifar10_hp_optimizer = PoutyneCifar10HpOptimizer()
    #
    #     cifar, cifar_test = load_cifar10()
    #     cifar.transform = ToTensor()
    #     cifar_test.transform = ToTensor()
    #
    #     param_gen = cifar10_hp_optimizer.optimize(param_gen, cifar[0], cifar[1])
    #     opt_hp = param_gen.get_best_param()
    #
    #     train_loader, valid_loader = train_valid_loaders(cifar, batch_size=int(opt_hp.get("batch_size")))
    #     test_loader = DataLoader(cifar_test, batch_size=int(opt_hp.get("batch_size")))
    #
    #     net = CifarNet()
    #     optimizer = optim.SGD(net.parameters(), lr=opt_hp.get("learning_rate"))
    #
    #     model = pt.Model(net, optimizer, 'cross_entropy', batch_metrics=['accuracy'])
    #     if torch.cuda.is_available():
    #         model.cuda()
    #
    #     history = model.fit_generator(train_loader, valid_loader, epochs=int(opt_hp.get("epoch")), verbose=True)
    #     test_loss, test_acc = model.evaluate_generator(test_loader)
    #
    #     self.assertTrue(test_acc >= 0.75, f"Random Gen result: {test_acc:.2f}%"
    #                                       f" in {time.time()-start_time:.2f} [s]")
    #     print(f"{opt_hp}, test_acc: {test_acc}")



if __name__ == '__main__':
    unittest.main()
