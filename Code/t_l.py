import seaborn as sn
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

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

    def plot(self, c_m, history, i=0):
        plt.figure(1 + i, figsize=(10, 7))
        sn.heatmap(c_m, annot=True, fmt='d', cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.figure(2 + i, figsize=(12, 9))
        plt.subplot(211)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'])

        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'])
        if i != 0:
            plt.show()
