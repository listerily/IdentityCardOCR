import cv2
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
        img_str = request.form['image']
        img_byte = base64.b64decode(img_str)
        image = np.fromstring(img_byte, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #print(f)
        # f.save('/var/www/uploads/uploaded_img.jpg')
        print("POST")
        return '上传成功'

    if request.method == 'GET':
        results = Driver(image,True)
        print("GET")
        # results = {
        #     'name': 'image_name',
        #     'nationality': 'image_nationality',
        #     'address': 'image_address',
        #     'number': 'image_number'
        # }

        # 显示结果页面
        return results



if __name__ == '__main__':
    app.run(port=8080)
