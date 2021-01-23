from modules.optimizer import HpOptimizer
from tests.pytorch_models import CifarNet
from tests.pytorch_datasets import train_valid_loaders, load_cifar10
import poutyne as pt
import time
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from typing import Dict, Union, List, Iterable, Hashable, Tuple
import numpy as np


class PoutyneCifar10HpOptimizer(HpOptimizer):
    def build_model(self, **hp) -> object:
        net = CifarNet()
        optimizer = optim.SGD(net.parameters(), lr=hp.get("learning_rate", 1e-3))
        model = pt.Model(net, optimizer, 'cross_entropy', batch_metrics=['accuracy'])
        model.cuda()
        return model

    def fit_model(
            self,
            model: pt.Model,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            **hp
    ) -> None:
        history = model.fit_generator(X, epochs=self.hp.get("epoch", 5), verbose=False)

    def score(
            self,
            model: pt.Model,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            **hp
    ) -> Tuple[float, float]:
        test_loss, test_acc = model.evaluate_generator(zip(X, y))
        return test_acc
