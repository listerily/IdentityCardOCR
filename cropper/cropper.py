import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


def read_and_preprocess(filename):
    image = cv2.imread(filename)
    image, alpha, beta = automatic_brightness_and_contrast(image, 4)
    image = np.clip((1.99 * image - 20), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    image = cv2.erode(image, np.ones((3, 3), np.uint8), iterations=1)
    image = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=1)
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
    crop(read_and_preprocess('/home/listerily/test3.png'), show=True, write=True)
    crop(read_and_preprocess('/home/listerily/test4.png'), show=True, write=True)
    crop(read_and_preprocess('/home/listerily/test5.png'), show=True, write=True)
    crop(read_and_preprocess('cards/0.png'), show=True, write=True)
