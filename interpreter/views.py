"""
Routes and views for the flask application.
"""
import os
from datetime import datetime
from flask import request
from flask import render_template
from interpreter import app
import requests
from interpreter.object_detection_iris import *



@app.route('/')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        year=datetime.now().year,
    )
    
#feel like the 304 might be b/c of the save_crops

@app.route('/imgurl', methods=['POST'])
def imgurl():
    imgpath = request.form['imgURL']
    if imgpath is None:
        return
    preds = processImgFromURL(imgpath)
    return render_template(
        'index.html',
        year=datetime.now().year,
        imgpath=imgpath,
        preds=preds[0],
        originalPreds=preds[1]
    )
