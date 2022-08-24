########################################################
#
#    MODULE TRAIN FIRSTNAME
#      TRAIN FIRSTNAME trains firstname characters
#    classifier using firstname datasets.
#
########################################################

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# Declaring chinese text font for matplotlib
plt.rcParams["font.sans-serif"] = ['SimHei']

AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 40
LEARNING_RATE = 0.001
MODEL_FILEPATH = '../saved_models/firstname_classifier'


# Model definition
class FirstNameClassifier(tf.keras.Model):
    def __init__(self, num_classes, name=None):
        super().__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(100, 100, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv4 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv5 = tf.keras.layers.Conv2D(512, kernel_size=3, activation='relu', padding='same')
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv6 = tf.keras.layers.Conv2D(1024, kernel_size=3, activation='relu', padding='same')
        self.pool6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = tf.keras.layers.Flatten()
        
        self.fc1 = tf.keras.layers.Dense(1024, activation='relu')
        self.fc2 = tf.keras.layers.Dense(1024, activation='relu')
        self.fc3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.pool6(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train(model):
    # ATTENTION: dataset file is not included in our submitted archive because they're too large.
    # Please generate them before your training.
    # Datasets could be generated using chinese_gen.py
    print('Training')
    data = np.load('dataset/firstname_0.npz')
    dataset = tf.data.Dataset.from_tensor_slices((data['arr_0'], data['arr_1']))
    dataset = dataset.shuffle(buffer_size=1000000)
    dataset = dataset.batch(batch_size=48)

    model.fit(dataset, epochs=EPOCHS)

    save(model)


def evaluate(model):
    print('Evaluating')
    data = np.load('dataset/firstname_1.npz')
    dataset = tf.data.Dataset.from_tensor_slices((data['arr_0'], data['arr_1']))
    dataset = dataset.shuffle(buffer_size=1000000)
    dataset = dataset.batch(batch_size=48)
    (loss, acc, a) = model.evaluate(dataset)
    print(loss, acc, a)


def save(model):
    model.save(MODEL_FILEPATH, save_format="tf")


def train_and_save():
    # Create model and train it.
    model = FirstNameClassifier(1000, name='firstname_classifier')
    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_crossentropy, tf.keras.metrics.sparse_categorical_accuracy]
    )
    train(model)

    # Load trained model and continue training.
    # model = tf.keras.models.load_model('../saved_models/firstname_classifier')
    # train(model)

    # Save and evaluation
    save(model)
    evaluate(model)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    train_and_save()
