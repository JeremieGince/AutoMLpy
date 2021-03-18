import tensorflow as tf
import tensorflow_datasets as tfds
from src.optimizers.optimizer import HpOptimizer
from typing import Union, Tuple
import numpy as np
import pandas as pd
from tests.tensorflow_items.tf_models import get_tf_mnist_model


class KerasMNISTHpOptimizer(HpOptimizer):
    def build_model(self, **hp) -> tf.keras.Model:
        model = get_tf_mnist_model(**hp)

        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=hp.get("learning_rate", 1e-3),
                nesterov=hp.get("nesterov", True),
                momentum=hp.get("momentum", 0.99),
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        return model

    def fit_dataset_model_(
            self,
            model: tf.keras.Model,
            dataset,
            **hp
    ) -> tf.keras.Model:
        history = model.fit(
            dataset,
            epochs=hp.get("epochs", 1),
            verbose=False,
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

    def score_on_dataset(
            self,
            model: tf.keras.Model,
            dataset,
            **hp
    ) -> Tuple[float, float]:
        test_loss, test_acc = model.evaluate(dataset, verbose=0)
        return test_acc, 0.0
