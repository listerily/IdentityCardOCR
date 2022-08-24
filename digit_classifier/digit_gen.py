########################################################
#
#    MODULE DIGIT DATASET GENERATOR
#      DIGIT DATASET GENERATOR generates digit images
#    to be used in training.
#    Randomized data augmentation would apply on all
#    images, including rotation, s&p noise, gaussian
#    noises, dilation, erosion, spots. Text size varies.
#
########################################################

import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import cv2
import random
import numpy as np
import pandas as pd


class DataAugmentation:
    def __init__(self, noise=True, dilate=True, erode=True):
        # Assign options
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    def add_salt_noise(self, img):
        # Salt and pepper noise. Randomly generates dot noises on the canvas.
        for i in range(random.randint(20, 25)):
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x, temp_y] = random.randint(50, 200)
        return img

    def add_gaussian_noise(self, img):
        # Gaussian noise background
        noise = np.random.normal(0, 1, img.shape) * random.randint(1, 5)
        return img + noise

    def add_erode(self, img):
        # Add erosion to the image. This makes text bolder.
        img = cv2.erode(img, np.ones((3, 3)))
        return img

    def add_dilate(self, img):
        # Add dilation to the image. This makes text slimmer.
        img = cv2.dilate(img, np.ones((3, 3)))
        return img

    def make_brighter(self, img):
        # Maker the whole image brighter.
        img = img + np.random.randint(30, 50, img.shape) * 1.
        return np.clip(img, 0, 255)

    def make_spots(self, img):
        # Randomly generate circle shaped noises on the canvas.
        origin_img = img
        for i in range(random.randint(1, 3)):
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img = cv2.circle(img, (temp_x, temp_y), 4, 230, -1)
            img = cv2.addWeighted(img, 0.5, origin_img, 0.5, 0)
        return np.clip(img, 0, 255)

    def do(self, img):
        # Automatic add noises using above functions.
        img = cv2.blur(img, (3, 3))
        if random.random() < 0.8:
            if self.dilate and random.random() < 0.5:
                img = self.add_dilate(img)
            else:
                img = self.add_erode(img)
        if random.random() < 0.5:
            img = self.add_gaussian_noise(img)
        if random.random() < 0.5:
            img = self.add_salt_noise(img)
            img = cv2.blur(img, (3, 3))
        if random.random() < 0.5:
            img = self.make_brighter(img)
        if random.random() < 0.5:
            img = self.make_spots(img)
            img = cv2.blur(img, (3, 3))
        return img


class DataGenerator:
    def __init__(self, character_filepath, font_filepath_set,
                 image_size=(44, 44), text_offset=(5, 1), font_size=range(35, 60),
                 debug=False):
        self.debug = debug
        # Initialize data generator.
        # Read character set from external file
        df = pd.read_csv(character_filepath)
        self.character_set = df['name'].tolist()
        # Read fonts from external file
        self.fonts = []
        for font_filepath in font_filepath_set:
            for f_size in font_size:
                # Add different sized font to fonts list
                font = ImageFont.truetype(font_filepath, f_size)
                self.fonts.append(font)
        # Text offset controls the position of every character on the canvas
        self.text_offset = text_offset
        # Size of the canvas
        self.image_size = image_size

    def generate(self, num, seed=None):
        if seed is not None:
            random.seed(seed)
        # Creating data augmentation object
        augmentation = DataAugmentation()
        # x: images, y: labels
        x = np.zeros((num, self.image_size[0], self.image_size[1], 1), dtype=np.float32)
        y = np.zeros((num,), dtype=np.uint8)
        for i in range(num):
            # Randomly choose a digit
            chi = random.randint(0, len(self.character_set) - 1)
            # Assign label
            y[i] = chi
            char = self.character_set[chi]
            # Randomly choose a font
            font = random.choice(self.fonts)
            # Randomly select a position based on text_offset
            random_offset = self.text_offset[0] + random.randint(-7, 7), self.text_offset[1] + random.randint(-7, 7)
            # Create new PIL grayscale image
            image = Image.new('L', self.image_size)
            d = ImageDraw.Draw(image)
            d.text(random_offset, char, font=font, fill=255)
            image = image.rotate(random.random() * 10 * random.choice([-1, 1]), expand=False)
            image = np.array(image)
            image = 255 - image  # Inverse black and white
            image = augmentation.do(image)  # Apply data augmentation
            image = image.astype(np.float32)  # convert to type float32
            # Plot images out if debug option is True
            if self.debug:
                plt.title('Label: ' + char + ', Index: ' + str(chi))
                plt.imshow(image, 'gray')
                plt.show()
            x[i, :, :, :] = np.expand_dims(image, axis=-1)
        return x, y


if __name__ == '__main__':
    # Code used for debugging
    generator = DataGenerator('digit.csv', ['OCR-B 10 BT.ttf'], debug=True)
    generator.generate(1)
