import json

import cv2
import flask
import numpy as np
from flask import Flask, request
from driver.driver import Driver
import base64

app = Flask(__name__)


@app.after_request
def cors(environ):
    environ.headers['Access-Contorl-Allow-Origin'] = '*'
    environ.headers['Access-Contorl-Allow-Method'] = '*'
    environ.headers['Access-Contorl-Allow-Headers'] = 'x-requested=with,content-type'
    return environ


@app.route('/api', methods=['POST','GET'], strict_slashes=False)
def index():
    image = None
    if request.method == 'POST':
        # data = json.loads(flask.request.get_data("data"))
        # data_64 = str.encode(data['data'])
        img_upload = request.files.get("img_upload")
        # 判断是否接收到图片
        if img_upload:
            # 读取图片
            image_string = base64.b64encode(img_upload.read())
            image_string = str(image_string, "utf8")
        # img_str = request.data['image']
        # img_byte = base64.b64decode(img_str)
        # image = np.fromstring(img_byte, np.uint8)
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #print(f)
        # f.save('/var/www/uploads/uploaded_img.jpg')
        print("上传")
        return '上传成功'

    if request.method == 'GET':
        results = Driver(image,True)
        # results = {
        #     'name': 'image_name',
        #     'nationality': 'image_nationality',
        #     'address': 'image_address',
        #     'number': 'image_number'
        # }
        print("获取")
        # 显示结果页面
        return results
    print("hello")
    return None



if __name__ == '__main__':
    app.run(host="localhost",port=8080)
