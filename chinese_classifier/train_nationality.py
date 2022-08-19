import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import random
from keras.layers import *
from chinese_gen import DataGenerator
plt.rcParams["font.sans-serif"] = ['SimHei']

AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 15
MODEL_FILEPATH = '../saved_models/chinese_nationality_classifier'


class VGG(tf.keras.Model):
    def __init__(self, num_classes, name=None):
        super().__init__(name=name)

        self.conv1a = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same',
                                             input_shape=(44, 44, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv2a = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv3a = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.flatten = tf.keras.layers.Flatten()

        self.fc7 = tf.keras.layers.Dense(256, activation='relu')
        self.fc8 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1a(inputs)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc7(x)
        x = self.fc8(x)
        return x


def generate_dataset(generator, num):
    data, label = generator.generate(num)
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(buffer_size=200000)
    dataset = dataset.batch(batch_size=128)
    return dataset.prefetch(AUTOTUNE)


def train(model, train_dataset):
    print('Training')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_crossentropy, tf.keras.metrics.sparse_categorical_accuracy]
    )

    model.fit(train_dataset, epochs=EPOCHS)


def evaluate(model, test_dataset):
    print('Evaluating')
    (loss, acc, a) = model.evaluate(test_dataset)
    print(loss, acc, a)


def save(model):
    model.save(MODEL_FILEPATH, save_format="tf")


def train_and_save():
    generator = DataGenerator('chinese_nationality.csv', ['STXihei.ttf'])
    model = VGG(87, name='VGG')
    train_dataset = generate_dataset(generator, 400) #每个字400张，汉另外加870
    test_dataset = generate_dataset(generator, 50)
    train(model, train_dataset)
    evaluate(model, test_dataset)
    save(model)


def continue_train():
    generator = DataGenerator('chinese_nationality.csv', ['STXihei.ttf'])
    model = tf.keras.models.load_model(MODEL_FILEPATH)
    train_dataset = generate_dataset(generator, 20)
    test_dataset = generate_dataset(generator, 8)
    train(model, train_dataset)
    evaluate(model, test_dataset)
    save(model)

def load_and_predict():
    model = tf.keras.models.load_model(MODEL_FILEPATH)

    image = cv2.imread('./%d.png' % 17)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (44, 44))
    image = image.astype(np.float32)
    # image = np.round(image)
    image = np.expand_dims(image, axis=-1)
    result = model.predict(np.array([image])).argmax()
    r = character_set[result]
    plt.title('Prediction Result: ' + str(r))
    plt.imshow(image, 'gray')
    plt.show()


def predict_some():
    model = tf.keras.models.load_model(MODEL_FILEPATH)
    generator = DataGenerator('chinese_nationality.csv', ['STXihei.ttf'])
    data, label = generator.generate(1)

    print(data.shape)
    print(label.shape)
    print(label)
    for i in range(30):
        image = random.choice(data)
        result = model.predict(np.array([image])).argmax()
        r = character_set[result]
        plt.title('Prediction Result: ' + r)
        plt.imshow(image, 'gray')
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv('chinese_nationality.csv', encoding='utf-8')
    character_set = df['name'].tolist()
    train_and_save()
    # for i in range(60):
    #     continue_train()
    # load_and_predict()
    # predict_some()
