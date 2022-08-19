from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageFont, ImageDraw
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置显示中文字体
plt.rcParams["font.sans-serif"] = ['SimHei']
NUM_PARALLEL = 2


class DataAugmentation:
    def __init__(self, noise=True, dilate=True, erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    def add_salt_noise(self, img):
        for i in range(random.randint(50, 60)):  # 噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x, temp_y] = random.randint(50, 200)
        return img

    def add_spotty_salt_noise(self, img):
        for i in range(random.randint(2, 4)):
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[max(0, temp_x - 1):min(temp_x + 1, img.shape[0]),
                max(0, temp_y - 1):min(temp_y + 1, img.shape[0])] = random.randint(50, 200)
        return img

    def add_gaussian_noise(self, img):
        noise = np.random.normal(0, 1, img.shape) * random.randint(1, 5)
        return img + noise

    def add_erode(self, img):  # 腐蚀
        img = cv2.erode(img, np.ones((5, 5)))
        return img

    def add_dilate(self, img):  # 膨胀
        img = cv2.dilate(img, np.ones((4, 4)))
        return img

    def make_brighter(self, img):
        img = img + np.random.randint(20, 40, img.shape) * 1.
        return np.clip(img, 0, 255)

    def make_spots(self, img):
        origin_img = img
        for i in range(random.randint(1, 3)):
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img = cv2.circle(img, (temp_x, temp_y), 3, 230, -1)
            img = cv2.addWeighted(img, 0.5, origin_img, 0.5, 0)
        return np.clip(img, 0, 255)

    def do(self, img):
        if self.dilate:
            if random.random() < 0.5:
                img = self.add_dilate(img)
        elif self.erode:
            if random.random() < 0.5:
                img = self.add_erode(img)
        img = self.add_salt_noise(img)
        img = self.add_spotty_salt_noise(img)
        img = cv2.blur(img, (2, 2))
        if random.random() < 0.5:
            img = self.make_spots(img)
            img = cv2.blur(img, (2, 2))
        return cv2.threshold(img, 229, 255, cv2.THRESH_BINARY)[1]


class DataGenerator:
    def __init__(self, character_filepath, font_filepath_set,
                 image_size=(100, 100), text_offset=(6, -20), font_size=range(90, 100),
                 debug=False):
        self.debug = debug

        df = pd.read_csv(character_filepath, encoding='utf-8')
        self.character_set = df['name'].tolist()
        self.character_len = len(self.character_set)
        self.fonts = []
        for font_filepath in font_filepath_set:
            for f_size in font_size:
                font = ImageFont.truetype(font_filepath, f_size)
                self.fonts.append(font)
        self.text_offset = text_offset
        self.image_size = image_size
        self.augmentation = DataAugmentation()

    def generate_slice(self):
        part_x = np.zeros((self.character_len, self.image_size[0], self.image_size[1], 1), dtype=np.float32)
        part_y = np.zeros((self.character_len,), dtype=np.int32)
        for j in range(self.character_len):
            font = random.choice(self.fonts)

            part_y[j] = j
            char = self.character_set[j]
            random_offset = self.text_offset[0] + random.randint(-2, 2), self.text_offset[1] + random.randint(-2, 2)

            image = Image.new('L', self.image_size)
            d = ImageDraw.Draw(image)
            d.text(random_offset, char, font=font, fill=255)
            image = image.rotate(random.random() * 5 * random.choice([-1, 1]), expand=False)
            image = np.array(image)

            image = 255 - image
            image = self.augmentation.do(image)
            image = image.astype(np.float32)
            if self.debug:
                plt.title('Label: ' + char + ', Index: ' + str(j))
                plt.imshow(image, 'gray')
                plt.show()
            part_x[j, :, :, :] = np.expand_dims(image, axis=-1)
        return part_x, part_y

    def generate(self, num, seed=None):
        if seed is not None:
            random.seed(seed)

        all_num = num * self.character_len
        data_x = np.zeros((all_num, self.image_size[0], self.image_size[1], 1), dtype=np.float32)
        data_y = np.zeros((all_num,), dtype=np.int32)
        for i in range(0, num):
            part_x, part_y = self.generate_slice()
            data_x[i * self.character_len:(i + 1) * self.character_len, :, :, :] = part_x
            data_y[i * self.character_len:(i + 1) * self.character_len] = part_y

        return data_x, data_y


def generate_sub_dataset(idx):
    generator = DataGenerator('firstname.csv', ['STXihei.ttf'], debug=False)
    x, y = generator.generate(100)
    np.savez('./dataset/firstname_%d' % idx, x, y)


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=NUM_PARALLEL) as executor:
        executor.map(generate_sub_dataset, range(4))
    # generator = DataGenerator('firstname.csv', ['STXihei.ttf'], debug=True)
    # generator.generate(2)
