import cv2
import matplotlib.pyplot as plt
from flask import Flask, request
from driver.driver import Driver
import base64
import numpy as np

app = Flask(__name__)


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
            results = Driver(image, True, True).run()
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
