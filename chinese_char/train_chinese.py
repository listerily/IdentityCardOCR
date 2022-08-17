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
    def __init__(self, num_classes, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.model = tf.keras.Sequential()

    def setModel(self):
        self.model.add(Conv2D(128, (3, 3), input_shape=(44, 44, 1)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())

        # Fully connected layer
        self.model.add(Dense(1024))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(self.num_classes))

        self.model.add(Activation('softmax'))
        return self.model


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

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy',
                                                  min_delta=0, patience=4, mode='max',
                                                  verbose=1, restore_best_weights=True)
    callbacks_list = [early_stop]

    model.fit(train_dataset, callbacks=callbacks_list, epochs=EPOCHS)


def evaluate(model, test_dataset):
    print('Evaluating')
    (loss, acc, a) = model.evaluate(test_dataset)
    print(loss, acc, a)


def save(model):
    model.save(MODEL_FILEPATH, save_format="tf")


def train_and_save():
    generator = DataGenerator('Chinese_labels.csv', ['STXihei.ttf', 'OCR-B 10 BT.ttf'])
    m = VGG(4956, name='VGG')
    m.setModel()
    model = m.model
    train_dataset = generate_dataset(generator, 15)
    test_dataset = generate_dataset(generator, 5)
    train(model, train_dataset)
    evaluate(model, test_dataset)
    save(model)


def continue_train():
    generator = DataGenerator('Chinese_labels.csv', ['STXihei.ttf', 'OCR-B 10 BT.ttf'])
    model = tf.keras.models.load_model(MODEL_FILEPATH)
    train_dataset = generate_dataset(generator, 15)
    test_dataset = generate_dataset(generator, 5)
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
    model = tf.keras.models.load_model(MODEL_FILEPATH)
    generator = DataGenerator('Chinese_labels.csv', ['STXihei.ttf', 'OCR-B 10 BT.ttf'])
    data, label = generator.generate(2)

    print(data.shape)
    print(label.shape)
    print(label)
    for image in data:
        result = model.predict(np.array([image])).argmax()
        r = character_set[result]

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
