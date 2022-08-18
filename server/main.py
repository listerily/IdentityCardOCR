import base64
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
import numpy as np
import cv2
app = Flask(__name__)


@app.route("/recognize", methods=['POST'])
def result():
    buffer = base64.b64decode(request.data)
    i = np.frombuffer(buffer, dtype=np.uint8)
    im = cv2.imdecode(i, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.show()
    return {
        'success': True,
        'number': '',
        'name': '',
        'nationality': ''
    }


if __name__ == "__main__":
    app.run()
