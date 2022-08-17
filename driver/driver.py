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
    def __init__(self, filepath, locate=True, debug=False):
        self.debug = debug
        self.locate = locate
        self.filepath = filepath
        self.image = cv2.imread(filepath)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

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
        # name_images = extract_characters(image_name, debug=self.debug)
        # nationality_images = extract_characters(image_nationality, debug=self.debug)
        # address_images = extract_characters(image_address, debug=self.debug)

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

        #Chinese Character Classification
        chinese_classifier=tf.keras.models.load_model('../saved_models/chinese_classifier')
        chinese_images=np.

if __name__ == '__main__':
    Driver('/home/listerily/IDCard/syx6.jpg', debug=True).run()
    # Driver('../../work/cards/0.png', locate=False, debug=True).run()
