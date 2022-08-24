########################################################
#
#    MODULE TEXT SEGMENTATOR
#      SEGMENTATOR separates cropped text image to
#    smaller images, each containing only one character.
#
########################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2


def segment_character(image, axis,
                      len_threshold=10,
                      space_density_threshold=4, character_density_threshold=8,
                      debug=False):
    # Segment characters based on pixel density
    # Calculate histogram on specific axis
    hist = np.sum(1 - image, axis=axis)
    if debug:
        plt.plot(np.arange(len(hist)), hist)
        plt.show()

    # Segmentation
    state = 0
    character_start_position = 0
    boxes = []
    for i in range(len_threshold, len(hist)):
        if state == 0 and hist[i - len_threshold:i].mean() >= character_density_threshold:
            # If pixel density is equal or above threshold then it might be a character
            state = 1
            character_start_position = i - len_threshold
        elif state == 1 and hist[i - len_threshold:i].mean() <= space_density_threshold:
            # If pixel density is equal or below threshold then it might be blank.
            state = 0
            boxes.append((character_start_position, i - len_threshold))
    if state == 1:
        # Append last box
        boxes.append((character_start_position, len(hist)))
    if debug:
        # Plot segment results if debug option is True
        image_copy = np.copy(image)
        t = image_copy.shape[axis]
        for box in boxes:
            image_copy = cv2.rectangle(image_copy,
                                       (box[0], 0) if axis == 0 else (0, box[0]),
                                       (box[1], t) if axis == 0 else (t, box[1]),
                                       (0, 0, 0), 2)
        plt.imshow(image_copy, 'gray')
        plt.show()
    return boxes


def extract_characters(image, debug=False, padding=3, area_threshold=1200):
    # Do segment over two axis and combine them together
    boxes = []
    # Segment over vertical
    v_boxes = segment_character(image, 1, debug=debug, space_density_threshold=8, character_density_threshold=10)
    for v_box in v_boxes:
        # Segment over horizontal
        h_boxes = segment_character(image[v_box[0]:v_box[1], :], 0, debug=debug)
        for h_box in h_boxes:
            if (h_box[1] - h_box[0]) * (v_box[1] - v_box[0]) <= area_threshold:
                continue
            # Add this box to boxes list
            boxes.append((max(0, h_box[0] - padding),
                          max(0, v_box[0] - padding),
                          min(image.shape[1], h_box[1] + 2 * padding),
                          min(image.shape[0], v_box[1] + 2 * padding)))

    if debug:
        # Plot segment results if debug mode turned on
        image_copy = np.copy(image)
        for box in boxes:
            image_copy = cv2.rectangle(image_copy,
                                       (box[0], box[1]),
                                       (box[2], box[3]),
                                       (0, 0, 0), 2)
        plt.title('Text Segmentation Result')
        plt.imshow(image_copy, 'gray')
        plt.show()
    return boxes
