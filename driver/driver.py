import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from locator.locator import locate_id_card, perspective_transform
from preprocessor.preprocess import preprocess, crop
from segmentator.number_segmentator import extract_numbers
from segmentator.text_segmentator import extract_characters


class Driver:
    def __init__(self, image, locate=True, debug=True):
        self.debug = debug
        self.locate = locate
        # self.filepath = filepath
        # self.image = cv2.imread(filepath)
        self.image = image
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        plt.title('Card Image')
        plt.imshow(self.image)
        plt.show()


    def run(self):
        if self.locate:
            points = locate_id_card(self.image, debug=self.debug)
            if points is None:
                return None

            transformed_image = perspective_transform(self.image, np.squeeze(points, axis=1))
            if self.debug:
                plt.title('Transformed Card Image')
                plt.imshow(transformed_image)
                plt.show()
        else:
            transformed_image = self.image

        preprocess_result = preprocess(transformed_image)

        if self.debug:
            plt.title('Preprocessed Image')
            plt.imshow(preprocess_result, 'gray')
            plt.show()

        image_name, image_nationality, image_address, image_number = crop(preprocess_result, debug=self.debug)

        # Number Segmentation
        image_number = cv2.morphologyEx(image_number, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)
        number_image_boxes = extract_numbers(image_number, debug=self.debug)

        # Text Segmentation
        image_name=cv2.morphologyEx(image_name, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)
        name_image_boxes = extract_characters(image_name, debug=self.debug)

        image_nationality=cv2.morphologyEx(image_nationality, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)
        nationality_image_boxes = extract_characters(image_nationality, debug=self.debug)

        image_address=cv2.morphologyEx(image_address, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)
        address_image_boxes = extract_characters(image_address, debug=self.debug)

        # Digit Classification
        digit_classifier = tf.keras.models.load_model('../saved_models/digit_classifier')
        digit_images = np.zeros((18, 44, 44, 1))
        for i, box in enumerate(number_image_boxes):
            digit_image = image_number[box[1]:box[3], box[0]:box[2]]
            desired_size = max(digit_image.shape[:2])
            digit_image = cv2.copyMakeBorder(digit_image,
                                             math.floor((desired_size - box[3] + box[1]) / 2),
                                             math.ceil((desired_size - box[3] + box[1]) / 2),
                                             math.floor((desired_size - box[2] + box[0]) / 2),
                                             math.ceil((desired_size - box[2] + box[0]) / 2),
                                             cv2.BORDER_CONSTANT, value=1.)
            digit_image = cv2.resize(digit_image, (44, 44)) * 255
            digit_image = digit_image.astype(np.float32)
            digit_image = np.expand_dims(digit_image, axis=-1)
            digit_images[i, :, :, :] = np.array([digit_image])
        digit_results = digit_classifier.predict(digit_images).argmax(axis=1)
        print(digit_results)

        # Classification
        chinese_classifier=tf.keras.models.load_model('../saved_models/chinese_classifier')

        name_images = np.zeros((len(name_image_boxes), 44, 44, 1))
        for i, box in enumerate(name_image_boxes):
            name_image = image_name[box[1]:box[3], box[0]:box[2]]
            desired_size = max(name_image.shape[:2])
            name_image = cv2.copyMakeBorder(name_image,
                                             math.floor((desired_size - box[3] + box[1]) / 2),
                                             math.ceil((desired_size - box[3] + box[1]) / 2),
                                             math.floor((desired_size - box[2] + box[0]) / 2),
                                             math.ceil((desired_size - box[2] + box[0]) / 2),
                                             cv2.BORDER_CONSTANT, value=1.)
            name_image = cv2.resize(name_image, (44, 44)) * 255
            name_image = name_image.astype(np.float32)
            name_image = np.expand_dims(name_image, axis=-1)
            name_images[i, :, :, :] = np.array([name_image])
        name_results = chinese_classifier.predict(name_images).argmax(axis=1)
        print(name_results)

        nationality_images = np.zeros((len(nationality_image_boxes), 44, 44, 1))
        for i, box in enumerate(nationality_image_boxes):
            nationality_image = image_nationality[box[1]:box[3], box[0]:box[2]]
            desired_size = max(nationality_image.shape[:2])
            nationality_image = cv2.copyMakeBorder(nationality_image,
                                             math.floor((desired_size - box[3] + box[1]) / 2),
                                             math.ceil((desired_size - box[3] + box[1]) / 2),
                                             math.floor((desired_size - box[2] + box[0]) / 2),
                                             math.ceil((desired_size - box[2] + box[0]) / 2),
                                             cv2.BORDER_CONSTANT, value=1.)
            nationality_image = cv2.resize(nationality_image, (44, 44)) * 255
            nationality_image = nationality_image.astype(np.float32)
            nationality_image = np.expand_dims(nationality_image, axis=-1)
            nationality_images[i, :, :, :] = np.array([nationality_image])
        nationality_results = chinese_classifier.predict(nationality_images).argmax(axis=1)
        print(nationality_results)

        address_images = np.zeros((len(address_image_boxes), 44, 44, 1))
        for i, box in enumerate(address_image_boxes):
            address_image = image_address[box[1]:box[3], box[0]:box[2]]
            desired_size = max(address_image.shape[:2])
            address_image = cv2.copyMakeBorder(address_image,
                                             math.floor((desired_size - box[3] + box[1]) / 2),
                                             math.ceil((desired_size - box[3] + box[1]) / 2),
                                             math.floor((desired_size - box[2] + box[0]) / 2),
                                             math.ceil((desired_size - box[2] + box[0]) / 2),
                                             cv2.BORDER_CONSTANT, value=1.)
            address_image = cv2.resize(address_image, (44, 44)) * 255
            address_image = address_image.astype(np.float32)
            address_image = np.expand_dims(address_image, axis=-1)
            address_images[i, :, :, :] = np.array([address_image])
        address_results = chinese_classifier.predict(address_images).argmax(axis=1)
        print(address_results)

        return {
            'number': digit_results,
            'name': name_results,
            'nationality': nationality_results,
            'address': address_results,
        }


if __name__ == '__main__':
    Driver(cv2.imread('/home/listerily/Desktop/syx6.jpg'), debug=True).run()
    # Driver('../../work/cards/0.png', locate=False, debug=True).run()
