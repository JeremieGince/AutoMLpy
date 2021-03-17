import tensorflow as tf
import tensorflow_datasets as tfds
from src.optimizers.optimizer import HpOptimizer
from typing import Union, Tuple
import numpy as np
import pandas as pd
from tests.tensorflow_items.tf_models import get_tf_mnist_model


class KerasMNISTHpOptimizer(HpOptimizer):
    def build_model(self, **hp) -> object:
        model = get_tf_mnist_model(**hp)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        return model

    def fit_model_(
            self,
            model: tf.keras.Model,
            X: Union[np.ndarray, pd.DataFrame, tf.Tensor],
            y: Union[np.ndarray, tf.Tensor],
            verbose=False,
            **hp
    ) -> object:
        history = model.fit(
            ds_train,
            epochs=hp.get("epochs", 1),
            verbose=verbose,
        )
        return model

    def score(
            self,
            model: tf.keras.Model,
            X: Union[np.ndarray, pd.DataFrame, tf.Tensor],
            y: Union[np.ndarray, tf.Tensor],
            **hp
    ) -> Tuple[float, float]:
        test_loss, test_acc = model
        return test_acc/100, 0.0
