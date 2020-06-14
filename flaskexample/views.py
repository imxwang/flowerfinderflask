"""
Routes and views for the flask application.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from datetime import datetime
from flask import request
from flask import render_template
from flaskexample import app
import requests
from flaskexample.object_detection_iris import *

module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
detector = hub.load(module_handle).signatures['default']
model = keras.models.load_model('./flaskexample/static/model4.h5', compile = False)


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )


@app.route('/imgurl', methods=['POST'])
def imgurl():
    imgpath = request.form['imgURL']
    downloaded_image_path = download_and_resize_image(imgpath)
    result = run_detector(detector = detector, path = downloaded_image_path)
    new_result = non_max_suppression(result)
    areas = get_box_areas(new_result)
    min_boxes = get_min_boxes(new_result, areas=areas, minsize=1.25)
    preds = crop_box(min_boxes, downloaded_image_path, directory='./flaskexample/instance/Crops', model = model) 
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
        imgpath=imgpath,
        predslength=len(preds),
        preds=preds
    )
