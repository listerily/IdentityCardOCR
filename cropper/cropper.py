import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


def read_and_preprocess(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha = 1.99
    beta = 50
    image = np.clip((alpha * image + beta), 0, 255).astype(np.uint8)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    image = cv2.resize(image, (1792, 1128))
    return image / 255.


def crop(image, show=True, write=False, write_directory='.'):
    roi = [
        ((290, 120), (800, 270), 'name'),
        ((670, 280), (920, 410), 'nationality'),
        ((290, 560), (1110, 900), 'address'),
        ((550, 910), (1620, 1090), 'number'),
    ]
    images = []
    for r in roi:
        sub_image = image[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        images.append(sub_image)
    if show:
        for r in roi:
            image = cv2.rectangle(image, r[0], r[1], (0, 0, 0), 2)
        plt.imshow(image, 'gray')
        plt.show()
    if write:
        for r in roi:
            sub_image = image[r[0][1]:r[1][1], r[0][0]:r[1][0]]
            Image.fromarray(np.uint8(sub_image.astype(float) * 255)).save(write_directory + '/' + r[2] + '.png')

    return images


if __name__ == '__main__':
    crop(read_and_preprocess('./cards/0.png'), show=False, write=True)
