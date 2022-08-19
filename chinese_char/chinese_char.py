from PIL import Image, ImageFont, ImageDraw
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 设置显示中文字体
plt.rcParams["font.sans-serif"] = ['SimHei']


class DataAugmentation:
    def __init__(self, noise=True, dilate=True, erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    def add_salt_noise(self, img):
        for i in range(random.randint(20, 25)):  # 噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x, temp_y] = random.randint(50, 200)
        return img

    def add_gaussian_noise(self, img):
        noise = np.random.normal(0, 1, img.shape) * random.randint(1, 5)
        return img + noise

    def add_erode(self, img):  # 腐蚀
        img = cv2.erode(img, np.ones((3, 3)))
        return img

    def add_dilate(self, img):  # 膨胀
        img = cv2.dilate(img, np.ones((3, 3)))
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
                 image_size=(44, 44), text_offset=(1, -5), font_size=range(35, 48),
                 debug=False):
        self.debug = debug

        df = pd.read_csv(character_filepath, encoding='utf-8')
        self.character_set = list(df['name'])
        self.character_len = len(self.character_set)
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
        all_num = (num + 10) * self.character_len
        x = np.zeros((all_num, self.image_size[0], self.image_size[1], 1), dtype=np.float32)
        y = np.zeros((all_num,), dtype=np.int32)
        for i in range(0, all_num, self.character_len):
            for j in range(self.character_len):
                k = j
                if i >= num * self.character_len:
                    k = 50
                y[i + j] = k
                char = self.character_set[k]
                font = random.choice(self.fonts)
                random_offset = self.text_offset[0] + random.randint(-3, 3), self.text_offset[1] + random.randint(-3, 3)

                image = Image.new('L', self.image_size)
                d = ImageDraw.Draw(image)
                d.text(random_offset, char, font=font, fill=255)
                image = image.rotate(random.random() * 10 * random.choice([-1, 1]), expand=False)
                image = np.array(image)

                image = 255 - image
                image = augmentation.do(image)
                image = image.astype(np.float32)
                if self.debug:
                    plt.title('Label: ' + char + ', Index: ' + str(k))
                    plt.imshow(image, 'gray')
                    plt.show()
                x[i + j, :, :, :] = np.expand_dims(image, axis=-1)
        return x, y


if __name__ == '__main__':
    generator = DataGenerator('chinese_nationality.csv', ['STXihei.ttf'], debug=True)
    x, y = generator.generate(1)
    print(y)