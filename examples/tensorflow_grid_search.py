# Some useful packages
from typing import Union, Tuple
import time
import numpy as np
import pandas as pd
import pprint

# Tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds

# Importing the HPOptimizer and the RandomHpSearch from the AutoMLpy package.
from AutoMLpy import HpOptimizer, GridHpSearch


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def get_tf_mnist_dataset(**kwargs):
    # https://www.tensorflow.org/datasets/keras_example
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # Build training pipeline
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Build evaluation pipeline
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test


def get_tf_mnist_model(**hp):

    if hp.get("use_conv", False):
        model = tf.keras.models.Sequential([
            # Convolution layers
            tf.keras.layers.Conv2D(10, 3, padding="same", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(50, 3, padding="same"),
            tf.keras.layers.MaxPool2D((2, 2)),

            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

    return model


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

    def score_on_dataset(
            self,
            model: tf.keras.Model,
            dataset,
            **hp
    ) -> float:
        test_loss, test_acc = model.evaluate(dataset, verbose=0)
        return test_acc


if __name__ == '__main__':
    mnist_train, mnist_test = get_tf_mnist_dataset()
    mnist_hp_optimizer = KerasMNISTHpOptimizer()

    hp_space = dict(
        epochs=list(range(1, 16)),
        learning_rate=np.linspace(1e-4, 1e-1, 50),
        nesterov=[True, False],
        momentum=np.linspace(0.01, 0.99, 50),
        use_conv=[True, False],
    )

    param_gen = GridHpSearch(hp_space, max_seconds=60 * 10, max_itr=100)

    save_kwargs = dict(
        save_name=f"tf_mnist_hp_opt",
        title="Grid search: MNIST",
    )

    param_gen = mnist_hp_optimizer.optimize_on_dataset(
        param_gen, mnist_train, save_kwargs=save_kwargs,
        stop_criterion=1.0,
    )

    opt_hp = param_gen.get_best_param()

    model = mnist_hp_optimizer.build_model(**opt_hp)
    mnist_hp_optimizer.fit_dataset_model_(
        model, mnist_train, **opt_hp
    )

    test_acc = mnist_hp_optimizer.score_on_dataset(
        model, mnist_test, **opt_hp
    )

    print(f"test_acc: {test_acc * 100:.3f}%")

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(opt_hp)

    fig = param_gen.write_optimization_to_html(show=True, dark_mode=True, **save_kwargs)