import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import *
from digit_gen import DataGenerator

AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 10
LEARNING_RATE = .001
MODEL_FILEPATH = '../saved_models/digit_classifier'


class LeNet5(tf.keras.Model):
    def __init__(self, num_classes, name=None):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='relu', padding='same',
                                            input_shape=(44, 44, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=5, strides=1, activation='relu', padding='valid')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=5, strides=1, activation='relu', padding='valid')
        self.pool3 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(512, activation='relu')
        self.fc3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def generate_dataset(generator, num):
    data, label = generator.generate(num)
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(buffer_size=200000)
    dataset = dataset.batch(batch_size=32)
    return dataset.prefetch(AUTOTUNE)


def train(model, train_dataset):
    print('Training')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_crossentropy, tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(train_dataset, epochs=EPOCHS)


def evaluate(model, test_dataset):
    print('Evaluating')
    model.evaluate(test_dataset)


def save(model):
    model.save(MODEL_FILEPATH, save_format="tf")


def train_and_save():
    generator = DataGenerator('digit.csv', ['OCR-B 10 BT.ttf'])
    model = LeNet5(11, name='lenet5')
    train_dataset = generate_dataset(generator, 90000)
    test_dataset = generate_dataset(generator, 4096)
    train(model, train_dataset)
    evaluate(model, test_dataset)
    save(model)


def load_and_predict():
    model = tf.keras.models.load_model(MODEL_FILEPATH)
    for i in range(18):
        image = cv2.imread(str(i) + '.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        image = np.round(image)
        image = np.expand_dims(image, axis=-1)
        result = model.predict(np.array([image])).argmax()

        plt.title('Prediction Result: ' + str(result))
        plt.imshow(image, 'gray')
        plt.show()


if __name__ == '__main__':
    # train_and_save()
    load_and_predict()
