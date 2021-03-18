# AutoMLpy
---------------------------------------------------------------------------

This package is an automatic machine learning module whose function is to optimize the hyper-parameters 
of an automatic learning model. 

In this package you can find: a grid search method, a random search algorithm and a Gaussian process search method. 
Everything is implemented to be compatible with the _Tensorflow_, _pyTorch_ and _sklearn_ libraries. 


# Installation

```
pip install something...
```

And that's it!

 
 ---------------------------------------------------------------------------
# Example - MNIST optimization with Tensorflow & Keras

Here you can see an example on how to optimize a model made with Tensorflow and Keras on the popular dataset MNIST.

## Imports

We start by importing some useful stuff.


```python
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
from src import HpOptimizer
from src import RandomHpSearch

```

## Dataset

Now we load the MNIST dataset in the tensorflow way.


```python
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
```

## Keras Model

Now we make a function that return a keras model given a set of hyper-parameters (hp).


```python
def get_tf_mnist_model(**hp):

    if hp.get("use_conv"):
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

```

## The Optimizer Model

It's time to implement the optimizer model. You juste have to implement the following methods: "build_model",
"fit_dataset_model_" and "score_on_dataset". Those methods must respect there signatur and output type. The objective here is to make the building, the training and the score phase depend on some hyper-parameters. So the optimizer can use those to find the best set of hp.


```python
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
    ) -> Tuple[float, float]:
        test_loss, test_acc = model.evaluate(dataset, verbose=0)
        return test_acc, 0.0

```

## Execution & Optimization

First thing after creating our classes is to load the dataset in memory.


```python
mnist_train, mnist_test = get_tf_mnist_dataset()
mnist_hp_optimizer = KerasMNISTHpOptimizer()
```

After you will defined you hyper-parameters space with a dictionary like this.


```python
hp_space = dict(
    epochs=list(range(1, 16)),
    learning_rate=np.linspace(1e-4, 1e-1, 50),
    nesterov=[True, False],
    momentum=np.linspace(0.01, 0.99, 50),
    use_conv=[True, False],
)
```

It's time to defined you hp search algorithm and give it your budget in time and iteration.


```python
param_gen = RandomHpSearch(hp_space, max_seconds=60*10, max_itr=100)
```

Finally, you start the optimization by giving your parameter generator to the optimize method. Note that the "stop_criterion" argument is to stop the optimization when the given score is reach. It's really useful to save some time.


```python
save_kwargs = dict(
    save_name=f"tf_mnist_hp_opt",
    title="Random search: MNIST",
)

param_gen = mnist_hp_optimizer.optimize_on_dataset(
    param_gen, mnist_train, save_kwargs=save_kwargs,
    stop_criterion=1.0,
)
```
    

## Testing

Now, you can test the optimized hyper-parameters by fitting again we the full train dataset. Yes with the full dataset, cause in the optimization phase a cross-validation is made which crop your train dataset by half. Plus, it's time to test the fitted model on the test dataset.


```python
opt_hp = param_gen.get_best_param()

model = mnist_hp_optimizer.build_model(**opt_hp)
mnist_hp_optimizer.fit_dataset_model_(
    model, mnist_train, **opt_hp
)

test_acc, _ = mnist_hp_optimizer.score_on_dataset(
    model, mnist_test, **opt_hp
)

print(f"test_acc: {test_acc*100:.3f}%")
```
    

The optimized hyper-parameters:


```python
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(opt_hp)
```
    

## Visualization

You can visualize the optimization with a interactive html file.


```python
fig = param_gen.write_optimization_to_html(show=True, **save_kwargs)
```
 
 ---------------------------------------------------------------------------
 # Other examples
 Examples on how to use this package are in the folder "./examples". There you can find the previous example 
 with _Tensorflow_ and an example with _pyTorch_.


