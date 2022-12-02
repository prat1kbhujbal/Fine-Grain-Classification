import os
import numpy as np
import tensorflow as tf
from simple_cnn import Simple_CNN
from t_l import TF

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    t_s = (224, 224)
    method = "tranfer_learning"
    training_direc = '../Data/training/training'
    test_direc = '../Data/validation/validation'
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        shear_range=0.2,
        rescale=1. / 255,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        training_direc,
        target_size=t_s,
        batch_size=16,
        shuffle=True,
        subset='training',
        class_mode='categorical')

    validation_generator = train_datagen.flow_from_directory(
        training_direc,
        target_size=t_s,
        batch_size=16,
        shuffle=True,
        subset='validation',
        class_mode='categorical')

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255, validation_split=0.25)

    test_generator = test_datagen.flow_from_directory(
        test_direc,
        target_size=t_s,
        batch_size=16,
        shuffle=False,
        class_mode='categorical')

    if method == "simple_cnn":
        model = Simple_CNN(10)
        model.optimizer(model)
        history = model.fit(train_generator,
                            epochs=20,
                            validation_data=(validation_generator))
        print("Evaluation on test data")
        model.evaluate(test_generator)
        y_predicted = model.predict(test_generator, verbose=0)
        y_predicted_labels = np.argmax(y_predicted, axis=1)
        c_m = tf.math.confusion_matrix(
            labels=test_generator.labels, predictions=y_predicted_labels)
        model.plot(c_m, history)
    elif method == "tranfer_learning":
        pretrained_model = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
        tl = TF(pretrained_model, 10)
        model = tl.model()
        tl.optimizer(model, lr=1e-3)
        history = model.fit(
            train_generator,
            epochs=5,
            validation_data=validation_generator)
        model.evaluate(test_generator)
        y_predicted = model.predict(test_generator, verbose=0)
        y_predicted_labels = [np.argmax(i) for i in y_predicted]
        c_m = tf.math.confusion_matrix(
            labels=test_generator.labels, predictions=y_predicted_labels)
        tl.plot(c_m, history)
        model.trainable = True
        tl.optimizer(model, lr=1e-4)
        history_ft = model.fit(
            train_generator,
            epochs=5,
            validation_data=validation_generator)
        model.evaluate(test_generator)
        y_predicted = model.predict(test_generator, verbose=0)
        y_predicted_labels = [np.argmax(i) for i in y_predicted]
        c_m = tf.math.confusion_matrix(
            labels=test_generator.labels, predictions=y_predicted_labels)
        tl.plot(c_m, history_ft, 3)


if __name__ == '__main__':
    main()
