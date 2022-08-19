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

app = Flask(__name__)


def driver(image, locate, debug):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.title('Card Image')
    plt.imshow(image)
    plt.show()

    if locate:
        points = locate_id_card(image, debug=debug)
        if points is None:
            return None

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
    image_number = cv2.morphologyEx(image_number, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)
    number_image_boxes = extract_numbers(image_number, debug=debug)

    # Text Segmentation
    image_name = cv2.morphologyEx(image_name, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)
    name_image_boxes = extract_characters(image_name, debug=debug)

    image_nationality = cv2.morphologyEx(image_nationality, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)
    nationality_image_boxes = extract_characters(image_nationality, debug=debug)

    # image_address=cv2.morphologyEx(image_address, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)
    # address_image_boxes = extract_characters(image_address, debug=self.debug)

    # Digit Classification
    digit_classifier = tf.keras.models.load_model('../saved_models/digit_classifier')
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
    digit_results = str(digit_classifier.predict(digit_images).argmax(axis=1))
    print(digit_results)

    # Classification
    chinese_name_classifier = tf.keras.models.load_model('../saved_models/firstname_classifier_200_epoch')
    chinese_nationality_classifier = tf.keras.models.load_model('../saved_models/chinese_nationality_classifier')
    # chinese_classifier=tf.keras.models.load_model('../saved_models/chinese_nationality_classifier')

    name_images = np.zeros((len(name_image_boxes), 44, 44, 1))
    df = pd.read_csv('../chinese_classifier/firstname.csv', encoding='utf-8')
    name_sets = df['name'].tolist()
    for i, box in enumerate(name_image_boxes):
        name_image = image_name[box[1]:box[3], box[0]:box[2]]
        desired_size = max(name_image.shape[:2])
        name_image = cv2.copyMakeBorder(name_image,
                                        math.floor((desired_size - box[3] + box[1]) / 2),
                                        math.ceil((desired_size - box[3] + box[1]) / 2),
                                        math.floor((desired_size - box[2] + box[0]) / 2),
                                        math.ceil((desired_size - box[2] + box[0]) / 2),
                                        cv2.BORDER_CONSTANT, value=1.)
        name_image = cv2.resize(name_image, (44, 44)) * 255
        name_image = name_image.astype(np.float32)
        name_image = np.expand_dims(name_image, axis=-1)
        name_images[i, :, :, :] = np.array([name_image])
    name_results = chinese_name_classifier.predict(name_images).argmax(axis=1)
    name = []
    for r in name_results:
        print(name_sets[r])
        name.append(name_sets[r])

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
    nationality = []
    for r in nationality_results:
        print(nationality_sets[r])
        nationality.append(nationality_sets[r])

    # address_images = np.zeros((len(address_image_boxes), 44, 44, 1))
    # for i, box in enumerate(address_image_boxes):
    #     address_image = image_address[box[1]:box[3], box[0]:box[2]]
    #     desired_size = max(address_image.shape[:2])
    #     address_image = cv2.copyMakeBorder(address_image,
    #                                      math.floor((desired_size - box[3] + box[1]) / 2),
    #                                      math.ceil((desired_size - box[3] + box[1]) / 2),
    #                                      math.floor((desired_size - box[2] + box[0]) / 2),
    #                                      math.ceil((desired_size - box[2] + box[0]) / 2),
    #                                      cv2.BORDER_CONSTANT, value=1.)
    #     address_image = cv2.resize(address_image, (44, 44)) * 255
    #     address_image = address_image.astype(np.float32)
    #     address_image = np.expand_dims(address_image, axis=-1)
    #     address_images[i, :, :, :] = np.array([address_image])
    # address_results = chinese_classifier.predict(address_images).argmax(axis=1)
    # print(address_results)
    legal_id = check_id_code(digit_results,True)
    if legal_id is not None:
        return {
            'success': 1,
            'number': digit_results,
            'year': legal_id.get('year'),
            'month': legal_id.get('month'),
            'ydate': legal_id.get('date'),
            'name': name,
            'nationality': nationality,
            'address': 'address_results'
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


@app.route('/api', methods=['POST', 'GET'], strict_slashes=False)
def index():
    results = None
    if request.method == 'POST':
        # data = json.loads(flask.request.get_data("data"))
        # data_64 = str.encode(data['data'])
        image_upload = request.data
        # 判断是否接收到图片
        # print(image_upload)
        if image_upload:

            image_buffer = base64.b64decode(image_upload)
            # print('image_byte for byte:',image_buffer)
            print("接收成功")
            image = np.frombuffer(image_buffer, dtype=np.uint8)
            # print('image for np :', image)

            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # print('image for cv :', image)
            # print(image)

            results = driver(image, True, True)

            print('result in post', results)
            print(results.get('number').__class__)
            print(results.get('name').__class__)
            print(results.get('nationality').__class__)
            print(results.get('address').__class__)
            if results is not None:
                return results
            else:
                return '识别失败'
        else:
            print("接收失败")
            return '接受失败'

    # if request.method == 'GET':
    #     # print('image',image)
    #     # results = Driver(image,True).run()
    #     # print("GET")
    #     # # results = {
    #     # #     'name': 'image_name',
    #     # #     'nationality': 'image_nationality',
    #     # #     'address': 'image_address',
    #     # #     'number': 'image_number'
    #     # # }
    #     # print("获取")
    #     # print(results)
    #     # # 显示结果页面
    #     return results
    # return '上传失败'


if __name__ == '__main__':
    app.run(host="localhost", port=8080)
