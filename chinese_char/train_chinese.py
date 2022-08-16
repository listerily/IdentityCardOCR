import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import *
import chinese_char

AUTOTUNE = tf.data.experimental.AUTOTUNE


# 2. 生成 dataset
def dealing(image):
    image_scale = []
    for img in image:
        image_resized = tf.image.convert_image_dtype(img, dtype=tf.float32)
        im = image_resized / 255.0
        image_scale.append(im)
    return image_scale


# 3. 生成cnn网络，完成训练
class Model:
    def __init__(self):
        self.num_epochs = 1000
        self.learning_rate = 0.001
        self.input_shape = (44, 44, 1)
        self.model = tf.keras.Sequential()


class MyModel(Model):
    def __init__(self, num):
        super().__init__()
        self.num_classes = num

    def setModel(self):
        self.model.add(Conv2D(128, (3, 3), input_shape=self.input_shape))
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

    def trainModel(self, train_data, tf_test_fea, tf_test_lab):
        self.model.summary()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )

        # checkpoint
        filepath = "the_best_model_plus.hdf5"
        # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join('./', filepath), monitor='val_accuracy',
                                                        verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        self.model.fit(train_data, callbacks=callbacks_list, epochs=self.num_epochs,
                       validation_data=(tf_test_fea, tf_test_lab))

    def saveModel(self):
        self.model.save("my_h5_model.hdf5", include_optimizer=False, save_format="h5")


if __name__ == '__main__':
    train_digit, test_digit = chinese_char.create_char()

    train_photos = np.array(list(train_digit.values())).reshape((-1, 44, 44))
    train_name_index = dict((name, index) for index, name in enumerate(train_digit.keys()))
    t1 = [train_name_index[name] for name in train_digit.keys()] * 100
    t1.sort()
    train_labels = np.array(t1)

    test_photos = np.array(list(test_digit.values())).reshape((-1, 44, 44))
    test_name_index = dict((name, index) for index, name in enumerate(test_digit.keys()))
    t2 = [test_name_index[name] for name in test_digit.keys()] * 24
    t2.sort()
    test_labels = np.array(t2)

    tf_train_photos = tf.convert_to_tensor(dealing(train_photos))
    tf_train_labels = tf.convert_to_tensor(train_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices((tf_train_photos, tf_train_labels))

    tf_test_feature = tf.convert_to_tensor(dealing(test_photos))
    tf_test_labels = tf.convert_to_tensor(test_labels)

    train_dataset = train_dataset.shuffle(buffer_size=1000000)
    train_dataset = train_dataset.batch(batch_size=64)
    train_dataset = train_dataset.prefetch(AUTOTUNE)

    mm = MyModel(len(test_digit.keys()))
    mm.setModel()
    mm.trainModel(train_dataset, tf_test_feature, tf_test_labels)
    #lenet.saveModel()