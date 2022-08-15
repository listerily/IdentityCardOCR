import matplotlib.pyplot as plt
import numpy as np
import cv2
from locator.locator import locate_id_card, perspective_transform
from preprocessor.preprocess import preprocess, crop
from segmentator.number_segmentator import extract_numbers
from segmentator.text_segmentator import extract_characters


class Driver:
    def __init__(self, filepath, debug=False):
        self.debug = debug
        self.filepath = filepath
        self.image = cv2.imread(filepath)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def run(self):
        points = locate_id_card(self.image, debug=self.debug)
        if points is None:
            return None

        transformed_image = perspective_transform(self.image, np.squeeze(points, axis=1))
        if self.debug:
            plt.title('Transformed Card Image')
            plt.imshow(transformed_image)
            plt.show()

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


if __name__ == '__main__':
    Driver('/home/listerily/IDCard/syx23.jpg', debug=True).run()
