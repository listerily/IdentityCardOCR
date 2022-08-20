from flask import Flask, request
import base64
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from locator.locator import locate_id_card, perspective_transform
from preprocessor.preprocess import preprocess, crop
from segmentator.number_segmentator import extract_numbers
from segmentator.text_segmentator import extract_characters
from check_code import check_id_code
from flask import jsonify

app = Flask(__name__)

digit_classifier = tf.keras.models.load_model('../saved_models/digit_classifier')
chinese_firstname_classifier = tf.keras.models.load_model('../saved_models/firstname_classifier')
chinese_lastname_classifier = tf.keras.models.load_model('../saved_models/lastname_classifier')
chinese_nationality_classifier = tf.keras.models.load_model('../saved_models/chinese_nationality_classifier')


def driver(image, locate, debug):
    print(image.shape)
    if debug:
        plt.title('Card Image')
        plt.imshow(image)
        plt.show()

    if locate:
        points = locate_id_card(image, debug=debug)
        if points is None:
            return {'success': 0}

        transformed_image = perspective_transform(image, np.squeeze(points, axis=1))
        if debug:
            plt.title('Transformed Card Image')
            plt.imshow(transformed_image)
            plt.show()
    else:
        transformed_image = image
    preprocess_result = preprocess(transformed_image)
    if debug:
        plt.title('Preprocessed Image')
        plt.imshow(preprocess_result, 'gray')
        plt.show()

    image_name, image_nationality, image_address, image_number = crop(preprocess_result, debug=debug)
    # Number Segmentation
    number_image_boxes = extract_numbers(image_number, debug=debug)
    # Text Segmentation
    name_image_boxes = extract_characters(image_name, debug=debug)
    nationality_image_boxes = extract_characters(image_nationality, debug=debug)

    # Digit Classification
    digit_images = np.zeros((18, 44, 44, 1))
    for i, box in enumerate(number_image_boxes):
        digit_image = image_number[box[1]:box[3], box[0]:box[2]]
        desired_size = max(digit_image.shape[:2])
        digit_image = cv2.copyMakeBorder(digit_image,
                                         math.floor((desired_size - box[3] + box[1]) / 2),
                                         math.ceil((desired_size - box[3] + box[1]) / 2),
                                         math.floor((desired_size - box[2] + box[0]) / 2),
                                         math.ceil((desired_size - box[2] + box[0]) / 2),
                                         cv2.BORDER_CONSTANT, value=1.)
        digit_image = cv2.resize(digit_image, (44, 44)) * 255
        digit_image = digit_image.astype(np.float32)
        digit_image = np.expand_dims(digit_image, axis=-1)
        digit_images[i, :, :, :] = np.array([digit_image])
    digit_results = digit_classifier.predict(digit_images).argmax(axis=1)
    id_code = ''
    for r in range(0, len(digit_results)):
        id_code += str(digit_results[r])

    # Classification
    name_images = np.zeros((len(name_image_boxes), 100, 100, 1))
    df_firstname = pd.read_csv('../chinese_classifier/firstname.csv', encoding='utf-8')
    df_lastname = pd.read_csv('../chinese_classifier/lastname.csv', encoding='utf-8')
    for i, box in enumerate(name_image_boxes):
        name_image = image_name[box[1]:box[3], box[0]:box[2]]
        desired_size = max(name_image.shape[:2])
        name_image = cv2.copyMakeBorder(name_image,
                                        math.floor((desired_size - box[3] + box[1]) / 2),
                                        math.ceil((desired_size - box[3] + box[1]) / 2),
                                        math.floor((desired_size - box[2] + box[0]) / 2),
                                        math.ceil((desired_size - box[2] + box[0]) / 2),
                                        cv2.BORDER_CONSTANT, value=1.)
        name_image = cv2.resize(name_image, (100, 100)) * 255
        name_image = name_image.astype(np.float32)
        name_image = np.expand_dims(name_image, axis=-1)
        name_images[i, :, :, :] = np.array([name_image])
    firstname_results = chinese_firstname_classifier.predict(name_images[1:]).argmax(axis=1)
    lastname_results = chinese_lastname_classifier.predict(name_images[:1]).argmax(axis=1)
    name = ''
    for r in lastname_results:
        ans = df_lastname['name'].tolist()[r]
        name += ans
    for r in firstname_results:
        ans = df_firstname['name'].tolist()[r]
        name += ans

    if len(nationality_image_boxes) != 0:
        nationality_images = np.zeros((len(nationality_image_boxes), 44, 44, 1))
        df = pd.read_csv('../chinese_classifier/chinese_nationality.csv', encoding='utf-8')
        nationality_sets = df['name'].tolist()
        for i, box in enumerate(nationality_image_boxes):
            nationality_image = image_nationality[box[1]:box[3], box[0]:box[2]]
            desired_size = max(nationality_image.shape[:2])
            nationality_image = cv2.copyMakeBorder(nationality_image,
                                                   math.floor((desired_size - box[3] + box[1]) / 2),
                                                   math.ceil((desired_size - box[3] + box[1]) / 2),
                                                   math.floor((desired_size - box[2] + box[0]) / 2),
                                                   math.ceil((desired_size - box[2] + box[0]) / 2),
                                                   cv2.BORDER_CONSTANT, value=1.)
            nationality_image = cv2.resize(nationality_image, (44, 44)) * 255
            nationality_image = nationality_image.astype(np.float32)
            nationality_image = np.expand_dims(nationality_image, axis=-1)
            nationality_images[i, :, :, :] = np.array([nationality_image])
        nationality_results = chinese_nationality_classifier.predict(nationality_images).argmax(axis=1)
        nationality = ''
        for r in nationality_results:
            nationality += nationality_sets[r]
    else:
        nationality = ''

    legal_id = check_id_code(id_code, True)
    if legal_id is not None:
        return {
            'success': 1,
            'number': id_code,
            'year': legal_id.get('year'),
            'month': legal_id.get('month'),
            'date': legal_id.get('date'),
            'name': name,
            'gender': '男' if int(id_code[-2]) % 2 == 1 else '女',
            'nationality': nationality,
            'address': 'address'
        }
    else:
        return {
            'success': 0
        }


@app.after_request
def cors(environ):
    environ.headers['Access-Contorl-Allow-Origin'] = '*'
    environ.headers['Access-Contorl-Allow-Method'] = '*'
    environ.headers['Access-Contorl-Allow-Headers'] = 'x-requested=with,content-type'
    return environ


@app.route('/api', methods=['POST'], strict_slashes=False)
def index():
    image_upload = request.data
    image_buffer = base64.b64decode(image_upload)
    image = np.frombuffer(image_buffer, dtype=np.uint8)

    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = driver(image, True, False)
    print(results)
    return jsonify(results)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
