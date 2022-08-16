from __future__ import print_function

import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import cv2
import random
import numpy as np
import pandas as pd


class DataAugmentation(object):
    def __init__(self, noise=True, dilate=True, erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    @classmethod
    def add_noise(cls, img):
        for i in range(random.randint(5, 20)):  # 噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x, temp_y] = random.randint(42, 255)
        return img

    @classmethod
    def add_erode(cls, img):  # 腐蚀
        img = cv2.erode(img, np.ones((3, 3)))
        return img

    @classmethod
    def add_dilate(cls, img):  # 膨胀
        img = cv2.dilate(img, np.ones((3, 3)))
        return img

    def do(self, img):
        if self.noise and random.random() < 0.8:
            img = self.add_noise(img)
        if random.random() < 0.8:
            if self.dilate and random.random() < 0.5:
                img = self.add_dilate(img)
            elif self.erode and random.random() < 0.5:
                img = self.add_erode(img)
        return img


class DataGenerator:
    def __init__(self, character_filepath, font_filepath_set,
                 image_size=(44, 44), text_offset=(5, 1), font_size=range(45, 65),
                 debug=False):
        self.debug = debug
        df = pd.read_csv(character_filepath)
        self.character_set = df['name'].tolist()
        self.fonts = []
        for font_filepath in font_filepath_set:
            for f_size in font_size:
                font = ImageFont.truetype(font_filepath, f_size)
                self.fonts.append(font)
        self.text_offset = text_offset
        self.image_size = image_size

    def generate(self, num, seed=None):
        if seed is not None:
            random.seed(seed)

        augmentation = DataAugmentation()
        x = np.zeros((num, self.image_size[0], self.image_size[1], 1), dtype=np.float32)
        y = np.zeros((num,), dtype=np.uint8)
        for i in range(num):
            chi = random.randint(0, len(self.character_set) - 1)
            y[i] = chi
            char = self.character_set[chi]
            font = random.choice(self.fonts)
            random_offset = self.text_offset[0] + random.randint(-3, 3), self.text_offset[1] + random.randint(-3, 3)

            image = Image.new('L', self.image_size)
            d = ImageDraw.Draw(image)
            d.text(random_offset, char, font=font, fill=255)
            image = image.rotate(random.randint(-8, 8), expand=False)
            image = np.array(image)

            augmentation.do(image)
            image = 255 - image
            image = image.astype(np.float32)
            if self.debug:
                plt.title('Label: ' + char + ', Index: ' + str(chi))
                plt.imshow(image, 'gray')
                plt.show()
            x[i, :, :, :] = np.expand_dims(image, axis=-1)
        return x, y


if __name__ == '__main__':
    generator = DataGenerator('digit.csv', ['OCR-B 10 BT.ttf'], debug=True)
    generator.generate(20)
