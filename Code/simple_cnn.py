import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt


class Simple_CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            32,
            kernel_size=5,
            strides=1,
            activation='relu',
            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(
            32,
            kernel_size=3,
            strides=1,
            activation='relu',
            padding='same')
        self.m1 = tf.keras.layers.MaxPool2D()
        self.conv3 = tf.keras.layers.Conv2D(
            64,
            kernel_size=3,
            activation='relu',
            padding='same')
        self.m2 = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.dense4 = tf.keras.layers.Dense(64, activation='relu')
        self.dense5 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.m1(out)
        out = self.conv3(out)
        out = self.m2(out)
        out = self.flatten(out)
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dense4(out)
        out = self.dense5(out)
        return out

    def optimizer(self, model):
        """Specifying a loss function, an optimizer, and metrics to monitor.

        Args:
            model : model

        Returns:
            _type_: Compiled model
        """
        return model.compile(
            optimizer='adam',
            loss="categorical_crossentropy",
            metrics=['accuracy'])

    def plot(self, c_m, history):
        plt.figure(1, figsize=(10, 7))
        sn.heatmap(c_m, annot=True, fmt='d', cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.figure(2, figsize=(12, 9))
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
        plt.show()
