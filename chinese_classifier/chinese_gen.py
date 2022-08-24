########################################################
#
#    MODULE CHINESE CHARACTER DATASET GENERATOR
#      CHINESE CHARACTER DATASET GENERATOR generates
#    chinese character images being used in training.
#    Randomized data augmentation would apply on all
#    images, including rotation, s&p noise, gaussian
#    noises, dilation, erosion, spots. Text size varies.
#
########################################################

from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageFont, ImageDraw
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Matplotlib option: show chinese characters
plt.rcParams["font.sans-serif"] = ['SimHei']
# Parameter used for ProcessPoll.
# If you assign NUM_PARALLEL = 2, MAKE SURE you have AT LEAST 16GiB RAM!
NUM_PARALLEL = 2


class DataAugmentation:
    def __init__(self, noise=True, dilate=True, erode=True):
        # Assign options
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    def add_salt_noise(self, img):
        # Salt and pepper noise. Randomly generates dot noises on the canvas.
        for i in range(random.randint(50, 60)):
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x, temp_y] = random.randint(50, 200)
        return img

    def add_spotty_salt_noise(self, img):
        # Salt and pepper noise but a bit larger.
        # Randomly generates block-sized noises on the canvas.
        for i in range(random.randint(2, 4)):
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[max(0, temp_x - 1):min(temp_x + 1, img.shape[0]),
                max(0, temp_y - 1):min(temp_y + 1, img.shape[0])] = random.randint(50, 200)
        return img

    def add_gaussian_noise(self, img):
        # Gaussian noise background
        noise = np.random.normal(0, 1, img.shape) * random.randint(1, 5)
        return img + noise

    def add_erode(self, img):
        # Add erosion to the image. This makes text bolder.
        img = cv2.erode(img, np.ones((5, 5)))
        return img

    def add_dilate(self, img):
        # Add dilation to the image. This makes text slimmer.
        img = cv2.dilate(img, np.ones((4, 4)))
        return img

    def make_brighter(self, img):
        # Maker the whole image brighter.
        img = img + np.random.randint(20, 40, img.shape) * 1.
        return np.clip(img, 0, 255)

    def make_spots(self, img):
        # Randomly generate circle shaped noises on the canvas.
        origin_img = img
        for i in range(random.randint(1, 3)):
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img = cv2.circle(img, (temp_x, temp_y), 3, 230, -1)
            img = cv2.addWeighted(img, 0.5, origin_img, 0.5, 0)
        return np.clip(img, 0, 255)

    def do(self, img):
        # Automatic add noises using above functions.
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
        # Before returning the final result, threshold this image.
        return cv2.threshold(img, 229, 255, cv2.THRESH_BINARY)[1]


class DataGenerator:
    def __init__(self, character_filepath, font_filepath_set,
                 image_size=(100, 100), text_offset=(6, -20), font_size=range(90, 100),
                 debug=False):
        self.debug = debug

        # Initialize data generator.
        # Read character set from external file
        df = pd.read_csv(character_filepath, encoding='utf-8')
        self.character_set = df['name'].tolist()
        self.character_len = len(self.character_set)
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
        # Data augmentation object
        self.augmentation = DataAugmentation()

    def generate_slice(self):
        # Function generate_slice generate all characters in character set only once.
        # Part X: images, Part Y: labels
        part_x = np.zeros((self.character_len, self.image_size[0], self.image_size[1], 1), dtype=np.float32)
        part_y = np.zeros((self.character_len,), dtype=np.int32)
        for j in range(self.character_len):
            # Randomly choose a font
            font = random.choice(self.fonts)
            # Assign label
            part_y[j] = j
            # Pick target character and randomly select a position based on text_offset
            char = self.character_set[j]
            random_offset = self.text_offset[0] + random.randint(-2, 2), self.text_offset[1] + random.randint(-2, 2)
            # Create new PIL grayscale image
            image = Image.new('L', self.image_size)
            d = ImageDraw.Draw(image)
            d.text(random_offset, char, font=font, fill=255)  # Draw text
            image = image.rotate(random.random() * 5 * random.choice([-1, 1]), expand=False)
            image = np.array(image)
            image = 255 - image  # Inverse black and white
            image = self.augmentation.do(image)  # Apply data augmentation
            image = image.astype(np.float32)  # convert to type float32
            # Plot images out if debug option is True
            if self.debug:
                plt.title('Label: ' + char + ', Index: ' + str(j))
                plt.imshow(image, 'gray')
                plt.show()
            part_x[j, :, :, :] = np.expand_dims(image, axis=-1)
        return part_x, part_y

    def generate(self, num, seed=None):
        # Function generate will generate images based on num.
        # Every character would have 'num' images generated.
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
    # Generate dataset and export all images to *.npz file.
    generator = DataGenerator('firstname.csv', ['STXihei.ttf'], debug=False)
    x, y = generator.generate(100)
    np.savez('./dataset/firstname_%d' % idx, x, y)


if __name__ == '__main__':
    # Generate datasets using parallel
    with ProcessPoolExecutor(max_workers=NUM_PARALLEL) as executor:
        executor.map(generate_sub_dataset, range(4))

    # Generate some datasets and plot them out. These code was used for debug only.
    # generator = DataGenerator('firstname.csv', ['STXihei.ttf'], debug=True)
    # generator.generate(2)
