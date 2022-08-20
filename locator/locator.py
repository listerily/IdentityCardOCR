import threading
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def locate(scale_ratio, image, debug):
    h, w = image.shape[:2]
    full_area = h * w * scale_ratio * scale_ratio
    scaled_image = cv.resize(image, (int(scale_ratio * w),
                                     int(scale_ratio * h)), interpolation=cv.INTER_AREA)
    # Gaussian blur
    gaussian_blured_image = cv.GaussianBlur(scaled_image, (5, 5), 0)
    median_image = cv.medianBlur(gaussian_blured_image, 5)
    blured_image = cv.bilateralFilter(median_image, 13, 15, 15)

    # Grayscale image
    gray_image = cv.cvtColor(blured_image, cv.COLOR_RGB2GRAY)

    # Canny edge detection
    canny = cv.Canny(gray_image, 40, 120)
    if debug:
        plt.title('Canny Edge Detection Results')
        plt.imshow(canny, 'gray')
        plt.show()

    # Thresholding
    is_success, binary_image = cv.threshold(canny, 60, 255, cv.THRESH_OTSU)
    binary_image = cv.dilate(binary_image, np.ones((3, 3)))

    # Obtain all contours
    contours, hierarchy = cv.findContours(binary_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda cnt: cv.contourArea(cnt), reverse=True)[:10]
    if debug:
        plt.title('Visualizing Top-10 Contours')
        contour_image = cv.drawContours(scaled_image, contours, -1, (0, 255, 0), 3)
        plt.imshow(contour_image, 'gray')
        plt.show()

    # Obtain card area
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.03 * peri, True)

        x, y, w, h = cv.boundingRect(c)
        ratio = w * 1.0 / h
        area = w * h
        if area / full_area > .3 and 1 < ratio < 2 and len(approx) == 4:
            if debug:
                plt.imshow(scaled_image)
                for p in approx:
                    plt.scatter(p[0][0], p[0][1], marker="x", s=100, linewidths=3, color="r", zorder=10)
                plt.show()
            return approx / scale_ratio


def locate_id_card(image, debug):
    pool = ThreadPoolExecutor()
    ratios = [0.8, 0.6, 0.4, 0.2]
    futures = []
    for ratio in ratios:
        future = pool.submit(locate, ratio, image, debug)
        futures.append(future)
    for future in futures:
        result = future.result()
        if result is not None:
            return result
    return None


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def perspective_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped
