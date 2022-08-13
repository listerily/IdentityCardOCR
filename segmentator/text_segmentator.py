import matplotlib.pyplot as plt
import numpy as np
import cv2


def segment_character(image, axis,
                      len_threshold=8,
                      space_density_threshold=4, character_density_threshold=8,
                      show=False):
    hist = np.sum(1 - image / 255., axis=axis)
    if show:
        plt.plot(np.arange(len(hist)), hist)
        plt.show()

    # Segmentation
    state = 0
    character_start_position = 0
    boxes = []
    for i in range(len_threshold, len(hist)):
        if state == 0 and hist[i - len_threshold:i].mean() >= character_density_threshold:
            state = 1
            character_start_position = i - len_threshold
        elif state == 1 and hist[i - len_threshold:i].mean() <= space_density_threshold:
            state = 0
            boxes.append((character_start_position, i - len_threshold))
    if state == 1:
        boxes.append((character_start_position, len(hist)))
    if show:
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


def extract_characters(image, show=False, padding=0):
    boxes = []
    v_boxes = segment_character(image, 1, show=show)
    for v_box in v_boxes:
        h_boxes = segment_character(image[v_box[0]:v_box[1], :], 0, show=show)
        for h_box in h_boxes:
            boxes.append((max(0, h_box[0] - padding),
                          max(0, v_box[0] - padding),
                          min(image.shape[1], h_box[1] + 2 * padding),
                          min(image.shape[0], v_box[1] + 2 * padding)))

    if show:
        image_copy = np.copy(image)
        for box in boxes:
            image_copy = cv2.rectangle(image_copy,
                                       (box[0], box[1]),
                                       (box[2], box[3]),
                                       (0, 0, 0), 2)
        plt.imshow(image_copy, 'gray')
        plt.show()
    return boxes


if __name__ == '__main__':
    full_image = cv2.imread('address.png', 0)
    boxes = extract_characters(full_image, show=True, padding=5)
