from flask import Flask, render_template, request, send_file
import mimetypes, os, uuid, time
from PIL import Image
import PIL.ImageOps
from roboflow import Roboflow

rf = Roboflow(api_key="8dasIsb8Hg2wSLkqOHd4")
project = rf.workspace("smoking-wchnj").project("senior_project-3ohay")
model = project.version(7).model

mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')
app = Flask(__name__)


def model_process(filename):
    # TODO: Change it to make YOLO output
    image = Image.open("static/images/{}".format(filename))
    output_img = format(uuid.uuid4()) + ".png"
    model_output = model.predict("static/images/{}".format(filename), confidence=40, overlap=30)
    model_output.save('static/images/{}'.format(output_img))
    # model.predict("test.jpg", confidence=40, overlap=30).save("prediction.jpg")
    # if image.mode == 'RGBA':
    #     r, g, b, a = image.split()
    #     rgb_image = Image.merge('RGB', (r, g, b))
    #     inverted_image = PIL.ImageOps.invert(rgb_image)
    #     r2, g2, b2 = inverted_image.split()
    #     final_transparent_image = Image.merge('RGBA', (r2, g2, b2, a))
    #     final_transparent_image.save('static/images/{}'.format(output_img))
    #
    # else:
    #     inverted_image = PIL.ImageOps.invert(image)
    #     inverted_image.save('static/images/{}'.format(output_img))

    return_obj = {
        "img_name" : "output_img/{}".format(output_img),

        "log_info" : model_output.json()['predictions']
    }
    return return_obj





@app.route('/')
def render_html():
    return render_template('index.html')

@app.route('/imgUpload', methods=['post'])
def img_process():
    files = request.files
    file = files.get('file').read()
    image_name= format(uuid.uuid4()) + '.png'
    with open(os.path.abspath(f'static/images/{image_name}'), 'wb') as f:
        f.write(file)
    return_obj = model_process(image_name)

    return return_obj

@app.route('/output_img/<path:path>')
def return_img(path):
    return app.send_static_file("images/{}".format(path))

if __name__ == '__main__':
    app.run()


