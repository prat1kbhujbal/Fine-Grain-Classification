import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import os
import seaborn as sn

class TF():
    trainable = False

    def __init__(self, model, num_classes) -> None:
        super().__init__()
        self.pretrained_model = hub.KerasLayer(model, trainable=self.trainable)
        if not self.trainable:
            self.pretrained_model.arguments = dict(batch_norm_momentum=0.997)
        self.num_classes = num_classes

    def model(self):
        model = tf.keras.Sequential([
            self.pretrained_model,
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        return model

    def optimizer(self, model, l_r):
        """Specifying a loss function, an optimizer, and metrics to monitor.

        Args:
            model : model

        Returns:
            _type_: Compiled model
        """
        return model.compile(
            optimizer=tf.keras.optimizers.Adam(l_r),
            loss="categorical_crossentropy",
            metrics=['accuracy'])

  