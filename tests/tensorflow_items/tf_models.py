import tensorflow as tf
import tensorflow_datasets as tfds


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
