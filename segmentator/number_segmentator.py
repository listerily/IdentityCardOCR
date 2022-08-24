########################################################
#
#    MODULE NUMBER SEGMENTATOR
#      SEGMENTATOR separates cropped id number image to
#    18 number images, each containing only one number.
#
########################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans


def extract_numbers(image, debug=True):
    # Calculate vertical and horizontal histogram
    hist_vertical = np.sum(1 - image, axis=1)
    hist_horizontal = np.sum(1 - image, axis=0)
    # Find text regions
    region_vertical, = np.where(hist_vertical > 80)
    region_horizontal, = np.where(hist_horizontal > 10)
    top, bottom = max(0, region_vertical[0] - 5), min(image.shape[0], region_vertical[-1] + 5)
    left, right = max(0, region_horizontal[0] - 10), min(image.shape[1], region_horizontal[-1] + 10)
    roi = image[top:bottom, left:right]
    # Segment numbers using K-Means
    # First, find out dark pixels
    pixels = np.where(roi < 0.75)
    # Eliminate second dimension of pixels, making pixels locating on a horizontal line
    pixels[0][:] = (bottom - top) / 2
    pixels = np.dstack((pixels[0], pixels[1]))[0]
    # Constructing K-Means object and fitting
    kmeans = KMeans(init='k-means++', n_clusters=18).fit(pixels)
    # Obtain centroids
    centroids = kmeans.cluster_centers_
    # Get boxes from k-means centroids
    boxes = [(left + int(round(c[1] - 27)), top, left + int(round(c[1] + 27)), bottom) for c in centroids]
    if debug:
        # Plot segment results if debug option is True
        plt.title('Number Image Horizontal Histogram')
        plt.plot(np.arange(len(hist_horizontal)), hist_horizontal)
        plt.show()

        image_copy = np.copy(image)
        for box in boxes:
            cv2.rectangle(image_copy,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 0, 0), 2)
        plt.title('Number Segmentation Result')
        plt.imshow(image_copy, 'gray')
        plt.scatter(left + centroids[:, 1], top + centroids[:, 0],
                    marker="x", s=50, linewidths=1, color="r", zorder=10)
        plt.show()
    # Sort boxes based on their position
    return sorted(boxes, key=lambda x: x[0])
