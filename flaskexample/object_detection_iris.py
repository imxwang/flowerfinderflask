#for loading model
import re
import os
from os import path
from os import listdir
from tensorflow import keras

# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub
from keras_preprocessing import image

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps


import cv2


# Print Tensorflow version
#print(tf.__version__)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" 
detector = hub.load(module_handle).signatures['default']
execution_path = os.getcwd()
model_path = os.path.join(execution_path, 'flaskexample', 'model', 'model4_with_sigmoid.h5')
model = keras.models.load_model(model_path)
folder = os.path.join(execution_path, 'flaskexample', 'static', 'crops')

def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)


def download_and_resize_image(url, new_width=1280, new_height=1280, display=False):
    img_original_path = os.path.join(os.getcwd(), 'flaskexample', 'static', 'original')
    _, filename = tempfile.mkstemp(dir=img_original_path, suffix='.jpg')
    response = urlopen(url)
    pil_image = Image.open(BytesIO(response.read()))
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=100, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    #print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):

    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image


#load an image
def load_image(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

#run the object detector and choose only flower boxes
def run_detector(detector, path):
  img = load_image(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  result = detector(converted_img)


  result = {key:value.numpy() for key,value in result.items()}
  
  #print(result)
  detection_boxes = []
  detection_scores = np.array([])
  detection_class_entities = np.array([])
  temp_box = np.array([])

  for i in range(0, len(result['detection_class_entities'])):
    temp = result['detection_class_entities'][i].decode("utf-8")

    if "Flower"==temp or "Rose"==temp or "Lily"==temp:
      detection_box = (result['detection_boxes'][i]).tolist()
      detection_boxes.append(detection_box)
      detection_scores = np.append(detection_scores, result['detection_scores'][i])
      detection_class_entities = np.append(detection_class_entities, result['detection_class_entities'][i])
  detection_boxes = np.array(detection_boxes)
  new_result = {'detection_boxes': detection_boxes, 'detection_scores': detection_scores, 'detection_class_entities': detection_class_entities}

  return new_result
  
  
#draw the boxes on the image
def initiate_all_boxes(new_result, path): 
  img = load_image(path)
  image_with_boxes = draw_boxes(
      img.numpy(), new_result["detection_boxes"],
      new_result["detection_class_entities"], new_result["detection_scores"])
  display_image(image_with_boxes)
 
#only select largest boxes with prediction threshold 
def non_max_suppression(result, iou_threshold=.1, score_threshold=.1):
  boxes = result['detection_boxes']
  scores = result['detection_scores']
  entities = result['detection_class_entities']

  selected_indices = tf.image.non_max_suppression(boxes = boxes, scores=scores, max_output_size=50, iou_threshold=iou_threshold, score_threshold = score_threshold)

  selected_boxes = tf.gather(boxes, selected_indices)
  selected_scores = tf.gather(scores, selected_indices)
  selected_entities = tf.gather(entities, selected_indices)
  new_result = {'detection_boxes': selected_boxes, 'detection_scores': selected_scores, 'detection_class_entities': selected_entities}
  return new_result

#get box areas
def get_box_areas(result):
  area = []
  for box in result['detection_boxes']:
    box = tuple(box)
    w = box[3]-box[1]+1
    h = box[2]-box[0]+1
    area.append(w*h)
  return area

#get the indexes for the top boxes
def get_top_index(areas_list, ntop=10):
  top = ntop*-1
  topn_index = sorted(range(len(areas_list)), key=lambda i: areas_list[i])[top:]
  return topn_index

#get the top boxes
def get_top_boxes(result, indexes):
  top_boxes = []
  for index in indexes:
    top_boxes.append(result['detection_boxes'][index])
  return top_boxes

#get boxes that meet area threshold
def get_min_boxes(result, areas, minsize=1.25):
  indexes = [i for i,v in enumerate(areas) if v >=minsize]
  min_boxes = []
  for index in indexes:
    min_boxes.append(result['detection_boxes'][index])
  return min_boxes
  
#crop boxes and run predictions on the crops
def crop_box(top_boxes, downloaded_image_path, directory, model):
  predsdict = {}
  
  filedir=downloaded_image_path.split('.')[0].split('/')[-1]
  for box in range(0, len(top_boxes)):
    image_array=load_image(downloaded_image_path).numpy()
    image=Image.fromarray(image_array).convert("RGB")
    im_width, im_height = image.size
    bbox = tuple(top_boxes[box])
    
    ymin = bbox[0]
    xmin = bbox[1]
    ymax = bbox[2]
    xmax = bbox[3]
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
    a,b,c,d = int(left) , int(right) , int(top) ,int(bottom)

    image_array = image_array[c:d,a:b]
    converted=tf.image.convert_image_dtype((cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)), tf.float32)[tf.newaxis, ...]
   
    full_filename = directory+"/{}_crop{}.jpg".format(filedir, box)
    cv2.imwrite(full_filename, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    filename = 'crops/'+full_filename.split('/')[-1]
    preds = predictions(converted, threshold=.3)
    predsdict[filename]=preds
  return predsdict
  
 #save the crops in a directory   
def save_crops(top_boxes, downloaded_image_path, directory):
    filedir = downloaded_image_path.split('.')[0].split('/')[-1]
    fileNames = []
    for box in range(0, len(top_boxes)):
        image_array = load_image(downloaded_image_path).numpy()
        image=Image.fromarray(image_array).convert("RGB")
        im_width, im_height = image.size
        bbox = tuple(top_boxes[box])

        ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]

        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)

        a,b,c,d = int(left) - 10 if int(left) > 10 else 0, int(right) + 10 if int(right) + 10 < im_width else im_width, int(top) - 10 if int(top) > 10 else 0, int(bottom) + 10 if int(bottom) + 10 < im_height else im_height

        image_array = image_array[c:d,a:b]
        newFileName = "{}_crop{}".format(filedir, box)
        cv2.imwrite(os.path.join(directory, newFileName  + ".jpg"), cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        fileNames.append(newFileName)
    return fileNames

#resize the crops
def resize(image_path):
  img = image.load_img(image_path, target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img /= 255
  return img

#get predictions from images
def predictions(image, threshold=.5):
    preds = model.predict_proba(image, verbose=1)  
    labels = ['Calla Lily', 'Dahlia', 'Daisy', 'Iris', 'Lily', 'Peony', 'Ranunculus', 'Rose', 'Sunflower', 'Tulip']
    #print(preds)
    img_label = []
    img_values = []
    for i in range(0, len(preds[[0]][0])):
        if preds[[0]][0][i]>threshold:
            img_values.append((preds[[0]][0][i], i))

    img_values = sorted(img_values, reverse=True)
    if len(img_values)==0:
        img_label.append('Undetermined')
    else:
        for i in range(0, len(img_values)):
            img_label.append(labels[img_values[i][1]])
            
    return img_label


#get predictions from a directory with a certain name, at a prediction threshold of 50% default
def get_predictions(directory, threshold=.5, fileNames = []):
  predsdict={}
  for filename in fileNames:
    filename = filename + '.jpg'
    path = os.path.join(directory, filename)
    pic = resize(path)
    preds = predictions(image = pic, threshold = threshold)
    predsdict['crops/' + filename]=preds
  return predsdict
  
  #get the information we need for our app
def processImgFromURL(imgpath):
    downloaded_image = download_and_resize_image(imgpath)
    originalPreds = predictions(resize(downloaded_image), threshold=.1)
    origPredsDict = {}
    origPath = os.path.basename(downloaded_image)
    origPredsDict['original/' + origPath] = originalPreds
    result = run_detector(detector = detector, path = downloaded_image)
    new_result = non_max_suppression(result)
    areas = get_box_areas(new_result)
    min_boxes = get_min_boxes(new_result, areas = areas, minsize = 1.25)
    fileNames = save_crops(min_boxes, downloaded_image_path = downloaded_image, directory = folder)
    predsDict = get_predictions(folder, .5, fileNames)
    return [predsDict, origPredsDict]  




