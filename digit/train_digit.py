
import os
import pathlib
import sys
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.layers import *
import digit_char

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
        self.num_epochs = 20
        self.learning_rate = 0.001
        self.input_shape = (44, 44, 3)
        self.model = tf.keras.Sequential()


class LeNet5(Model):
    def __init__(self, num):
        super().__init__()
        self.num_classes = num

    def setModel(self):
        self.model.add(Conv2D(6, (5, 5), activation='relu', input_shape=self.input_shape))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Conv2D(16, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def trainModel(self, train_data, tf_test_fea, tf_test_lab):
        self.model.summary()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=4, mode='max', verbose=1,
                                                      restore_best_weights=True)
        # checkpoint
        filepath = "the_best_model.hdf5"
        # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join('./', filepath), monitor='val_accuracy',
                                                        verbose=1, save_best_only=True, mode='max')
        callbacks_list = [early_stop, checkpoint]

        self.model.fit(train_data, callbacks=callbacks_list, epochs=self.num_epochs, validation_data=(tf_test_fea, tf_test_lab))


def update_model():
    train_digit, test_digit = digit_char.create_digit()
    num = len(test_digit.keys())

    train_photos = np.array(list(train_digit.values())).reshape((-1, 44, 44, 3))
    # for i in range(train_photos.shape[0]):
    #     char_dir = os.path.join('./dataset/trains', "%0.5d" % 2)
    #
    #     if not os.path.isdir(char_dir):
    #         os.makedirs(char_dir)
    #     path_image = os.path.join(char_dir, "%0.4d.png" % i)
    #     cv2.imwrite(path_image, train_photos[i])
    state = np.random.get_state()
    np.random.shuffle(train_photos)
    np.random.set_state(state)
    # for i in range(train_photos.shape[0]):
    #     char_dir = os.path.join('./dataset/trains', "%0.5d" % 1)
    #
    #     if not os.path.isdir(char_dir):
    #         os.makedirs(char_dir)
    #     path_image = os.path.join(char_dir, "%0.4d.png" % i)
    #     cv2.imwrite(path_image, train_photos[i])

    train_name_index = dict((name, index) for index, name in enumerate(train_digit.keys()))
    t1 = [train_name_index[name] for name in train_digit.keys()] * int((train_photos.shape[0] / num))
    t1.sort()
    train_labels = np.array(t1)
    np.random.shuffle(train_labels)

    # print(train_labels)

    test_photos = np.array(list(test_digit.values())).reshape((-1, 44, 44, 3))
    test_name_index = dict((name, index) for index, name in enumerate(test_digit.keys()))
    t2 = [test_name_index[name] for name in test_digit.keys()] * int((test_photos.shape[0] / num))
    t2.sort()
    test_labels = np.array(t2)

    tf_train_photos = tf.convert_to_tensor(dealing(train_photos))
    tf_train_labels = tf.convert_to_tensor(train_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices((tf_train_photos, tf_train_labels))

    tf_test_feature = tf.convert_to_tensor(dealing(test_photos))
    tf_test_labels = tf.convert_to_tensor(test_labels)

    train_dataset = train_dataset.shuffle(buffer_size=2000000)
    train_dataset = train_dataset.batch(batch_size=32)
    train_dataset = train_dataset.prefetch(AUTOTUNE)

    lenet = LeNet5(num)
    lenet.setModel()
    lenet.trainModel(train_dataset, tf_test_feature, tf_test_labels)


if __name__ == '__main__':

    # update_model()

    model = tf.keras.models.load_model('./the_best_model.hdf5')
    model.summary()

    file_path = 'digits'
    root_dir = os.getcwd() + os.sep + file_path
    root_dir_path = pathlib.Path(root_dir)

    # image = cv2.imread('./digits/0.png', cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    # image_resized = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # img = tf.expand_dims(image_resized,axis=0)
    # print(img.shape)
    # im = image_resized / 255.0
    # predict = model.predict(img)
    # pre_y = np.argmax(predict)
    # print(pre_y)

    all_image_filename = [str(jpg_path) for jpg_path in root_dir_path.glob('**/*.png')]
    pre_list = []
    for f in all_image_filename:
        image = tf.keras.preprocessing.image.load_img(f)
        #image.show()
        img_tensor = tf.keras.preprocessing.image.img_to_array(img=image)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        predict = model.predict(img_tensor,batch_size=32)
        pre_y = np.argmax(predict)
        pre_list.append(pre_y)
    print(pre_list)


import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import *
import digit_char

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
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.input_shape = (44, 44, 1)
        self.model = tf.keras.Sequential()


class LeNet5(Model):
    def __init__(self, num):
        super().__init__()
        self.num_classes = num

    def setModel(self):
        self.model.add(Conv2D(6, (5, 5), activation='relu', input_shape=self.input_shape))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Conv2D(16, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def trainModel(self, train_data, tf_test_fea, tf_test_lab):
        self.model.summary()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )

        # checkpoint
        filepath = "../saved_models/digit_classifier.hdf5"
        # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join('./', filepath), monitor='val_accuracy',
                                                        verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        self.model.fit(train_data, callbacks=callbacks_list, epochs=self.num_epochs,
                       validation_data=(tf_test_fea, tf_test_lab))

    def saveModel(self):
        self.model.save("saved_models/digit_classifier.hdf5", include_optimizer=False, save_format="h5")


if __name__ == '__main__':
    train_digit, test_digit = digit_char.create_digit()

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

    train_dataset = train_dataset.shuffle(buffer_size=2000000)
    train_dataset = train_dataset.batch(batch_size=32)
    train_dataset = train_dataset.prefetch(AUTOTUNE)

    lenet = LeNet5(len(test_digit.keys()))
    lenet.setModel()
    lenet.trainModel(train_dataset, tf_test_feature, tf_test_labels)
    # lenet.saveModel()
