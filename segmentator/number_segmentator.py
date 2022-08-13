import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans


def extract_numbers(image, show=True):
    hist_vertical = np.sum(1 - full_image / 255., axis=1)
    hist_horizontal = np.sum(1 - full_image / 255., axis=0)
    region_vertical, = np.where(hist_vertical > 80)
    region_horizontal, = np.where(hist_horizontal > 10)
    top, bottom = max(0, region_vertical[0] - 5), min(image.shape[0], region_vertical[-1] + 5)
    left, right = max(0, region_horizontal[0] - 10), min(image.shape[1], region_horizontal[-1] + 10)
    roi = image[top:bottom, left:right]
    pixels = np.where(roi < 127)
    pixels[0][:] = (bottom - top) / 2
    pixels = np.dstack((pixels[0], pixels[1]))[0]
    kmeans = KMeans(init='k-means++', n_clusters=18).fit(pixels)
    centroids = kmeans.cluster_centers_
    boxes = [(left + int(round(c[1] - 30)), top, left + int(round(c[1] + 30)), bottom) for c in centroids]
    if show:
        image_copy = np.copy(image)
        for box in boxes:
            cv2.rectangle(image_copy,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 0, 0), 2)
        plt.imshow(image_copy, 'gray')
        plt.scatter(left + centroids[:, 1], top + centroids[:, 0],
                    marker="x", s=50, linewidths=1, color="r", zorder=10)
        plt.show()
    return boxes


if __name__ == '__main__':
    full_image = cv2.imread('number.png', 0)
    extract_numbers(full_image)
