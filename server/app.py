from flask import Flask, request, render_template
from driver.driver import Driver

app = Flask(__name__)


@app.after_request
def cors(environ):
    environ.headers['Access-Contorl-Allow-Origin'] = '*'
    environ.headers['Access-Contorl-Allow-Method'] = '*'
    environ.headers['Access-Contorl-Allow-Headers'] = 'x-requested=with,content-type'
    return environ

@app.route('/')
def hello_world():  # put application's code here

    return 'Hello World!'


@app.route('/upload_img', methods=['POST', 'GET'], strict_slashes=False)
def upload_img():
    return render_template('upload_img.html',)


@app.route('/result', methods=['POST','GET'], strict_slashes=False)
def get_results():
    if request.method == 'POST':
        f = request.files['image']
        print(f)
        f.save('/var/www/uploads/uploaded_img.jpg')
        #获得结果集results
        results = Driver('/var/www/uploads/uploaded_img.jpg',True)
        # results = {
        #     'name': 'image_name',
        #     'nationality': 'image_nationality',
        #     'address': 'image_address',
        #     'number': 'image_number'
        # }
        # 显示结果页面
        return render_template('result.html',**results)



if __name__ == '__main__':
    app.run()
