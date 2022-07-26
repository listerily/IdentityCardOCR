########################################################
#
#    MODULE PREPROCESSOR
#      PREPROCESSOR preprocess image by thresholding
#    and removing noises. Then preprocessor would crop
#    name area, nationality area, id number area into
#    new images.
#
########################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2


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


def preprocess(image):
    # Adjust brightness and contrast
    image, alpha, beta = automatic_brightness_and_contrast(image, 4)
    # Clip color value
    image = np.clip((1.99 * image - 20), 0, 255).astype(np.uint8)
    # Convert colorspace to GRAY
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Adaptive thresholding
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Resize image to a fixed size
    image = cv2.resize(image, (1792, 1128))
    # Binary thresholding
    _, image = cv2.threshold(image, 229, 255, cv2.THRESH_BINARY)
    return image / 255.


def crop(image, debug=True):
    # Define Range of Interest. We would crop datas using these positions
    roi = [
        ((290, 120), (800, 270), 'name'),
        ((670, 280), (920, 410), 'nationality'),
        ((290, 560), (1110, 900), 'address'),
        ((550, 910), (1620, 1090), 'number'),
    ]
    # Cropping data from image
    images = []
    for r in roi:
        sub_image = image[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        images.append(sub_image)
    if debug:
        # Plotting cropping boundaries if debug mode is on
        image_copy = np.copy(image)
        for r in roi:
            image_copy = cv2.rectangle(image_copy, r[0], r[1], (0, 0, 0), 2)
        plt.title('Cropping Boundaries')
        plt.imshow(image_copy, 'gray')
        plt.show()
    return images
