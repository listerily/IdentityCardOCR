import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.layers import *
from chinese_char import DataGenerator
plt.rcParams["font.sans-serif"] = ['SimHei']

AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_FILEPATH = '../saved_models/chinese_classifier'


class VGG(tf.keras.Model):
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

            self.conv4a = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, activation='relu', padding='same')
            self.conv4b = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, activation='relu', padding='same')
            self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

            self.flatten = tf.keras.layers.Flatten()

            self.fc7 = tf.keras.layers.Dense(1024, activation='relu')
            self.fc8 = tf.keras.layers.Dense(num_classes, activation='softmax')

        def call(self, inputs):
            x = self.conv1a(inputs)
            x = self.pool1(x)

            x = self.conv2a(x)
            x = self.pool2(x)

            x = self.conv3a(x)
            x = self.pool3(x)

            x = self.conv4a(x)
            x = self.conv4b(x)
            x = self.pool4(x)

            x = self.flatten(x)

            x = self.fc7(x)
            x = self.fc8(x)
            return x


def generate_dataset(generator, num):
    data, label = generator.generate(num)
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(buffer_size=200000)
    dataset = dataset.batch(batch_size=64)
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
    (loss, acc, a) = model.evaluate(test_dataset)
    print(loss, acc, a)


def save(model):
    model.save(MODEL_FILEPATH, save_format="tf")


def train_and_save():
    generator = DataGenerator('Chinese_labels.csv', ['STXihei.ttf'])
    model = VGG(4956, name='VGG')
    train_dataset = generate_dataset(generator, 20)
    test_dataset = generate_dataset(generator, 8)
    train(model, train_dataset)
    evaluate(model, test_dataset)
    save(model)


def continue_train():
    generator = DataGenerator('Chinese_labels.csv', ['STXihei.ttf'])
    model = tf.keras.models.load_model(MODEL_FILEPATH)
    train_dataset = generate_dataset(generator, 20)
    test_dataset = generate_dataset(generator, 8)
    train(model, train_dataset)
    evaluate(model, test_dataset)
    save(model)

# def load_and_predict():
#     model = tf.keras.models.load_model(MODEL_FILEPATH)
#
#     for i in range(18):
#         image = cv2.imread('./digits/%d.png' % i)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image = image.astype(np.float32)
#         # image = np.round(image)
#         image = np.expand_dims(image, axis=-1)
#         result = model.predict(np.array([image])).argmax()
#
#         plt.title('Prediction Result: ' + str(result))
#         plt.imshow(image, 'gray')
#         plt.show()


def predict_some():
    # model = tf.keras.models.load_model(MODEL_FILEPATH)
    generator = DataGenerator('Chinese_labels.csv', ['STXihei.ttf'])
    data, label = generator.generate(2)

    print(data.shape)
    print(label.shape)
    print(label)
    for i in range(len(label)):
        # result = model.predict(np.array([image])).argmax()
        # r = character_set[result]
        if i > 4945:
            image = data[i]
            plt.title('Prediction Result: ' + str(1))
            plt.imshow(image, 'gray')
            plt.show()


if __name__ == '__main__':
    df = pd.read_csv('Chinese_labels.csv', encoding='utf-8')
    character_set = df['Character'].tolist()
    train_and_save()
    for i in range(60):
        continue_train()
    # load_and_predict()
    # predict_some()
