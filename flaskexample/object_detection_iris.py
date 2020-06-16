# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

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

# For measuring the inference time.
import time
import cv2

#for loading model
import re
from os import listdir
from os.path import isfile, join
from tensorflow import keras

# Print Tensorflow version
#print(tf.__version__)

# Check available GPU devices.
#print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

"""## Example use

### Helper functions for downloading images and for visualization.

Visualization code adapted from [TF object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py) for the simplest required functionality.
"""

def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)


def download_and_resize_image(url, new_width=1280, new_height=1280, display=False):
    _, filename = tempfile.mkstemp(dir="./flaskexample/instance/Original", suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    #im_width, im_height = pil_image.size
    #new_height = int((im_height*1500)/im_width)
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

"""## Apply module

Load a public image from Open Images v4, save locally, and display.
"""

#image_url = "https://cdn.shopify.com/s/files/1/1204/3320/products/40558696243_e763056082_o_800x800.jpg?v=1554306597"  #@param
#downloaded_image_path = download_and_resize_image(image_url, 1280, 1280, True)

"""Pick an object detection module and apply on the downloaded image. Modules:
* **FasterRCNN+InceptionResNet V2**: high accuracy,
* **ssd+mobilenet V2**: small and fast.
"""

#module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]

#detector = hub.load(module_handle).signatures['default']

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def run_detector(detector, path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}
  
  #print(result)
  detection_boxes = []
  detection_scores = np.array([])
  detection_class_entities = np.array([])
  temp_box = np.array([])

  for i in range(0, len(result['detection_class_entities'])):
    temp = result['detection_class_entities'][i].decode("utf-8")
    #print(temp)
    if "Flower"==temp or "Rose"==temp or "Lily"==temp:
      #print(type(result['detection_boxes'][1]))
      #print(i, result['detection_boxes'][i])
      detection_box = (result['detection_boxes'][i]).tolist()
      detection_boxes.append(detection_box)
      detection_scores = np.append(detection_scores, result['detection_scores'][i])
      detection_class_entities = np.append(detection_class_entities, result['detection_class_entities'][i])
  detection_boxes = np.array(detection_boxes)
  new_result = {'detection_boxes': detection_boxes, 'detection_scores': detection_scores, 'detection_class_entities': detection_class_entities}

  #print("Found %d flowers." % len(new_result["detection_scores"]))
  #print("Inference time: ", end_time-start_time)
  return new_result

def initiate_all_boxes(new_result, path): 
  img = load_img(path)
  #converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  image_with_boxes = draw_boxes(
      img.numpy(), new_result["detection_boxes"],
      new_result["detection_class_entities"], new_result["detection_scores"])
  display_image(image_with_boxes)
  #plot = display_image(image_with_boxes)
  #plt.savefig('/content/drive/My Drive/Flowers insight project/testplot.png')

#result=run_detector(detector, downloaded_image_path)
#initiate_all_boxes(result, downloaded_image_path)

#print(result)
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

#new_result = non_max_suppression(result)

def get_box_areas(result):
  area = []
  for box in result['detection_boxes']:
    box = tuple(box)
    w = box[3]-box[1]+1
    h = box[2]-box[0]+1
    area.append(w*h)
  return area

# areas = get_box_areas(new_result)
# print(areas)

def get_top_index(areas_list, ntop=10):
  top = ntop*-1
  topn_index = sorted(range(len(areas_list)), key=lambda i: areas_list[i])[top:]
  return topn_index

# topn_index = get_top_index(areas)
# print(topn_index)
# fake_list = [52, 10, 11, 23, 14, 2]
# get_top_index(fake_list, ntop=1)

def get_top_boxes(result, indexes):
  top_boxes = []
  for index in indexes:
    top_boxes.append(result['detection_boxes'][index])
  return top_boxes

# top_boxes = get_top_boxes(new_result, topn_index)

# print(top_boxes)

# for box in top_boxes:
#   w = box[3]-box[1]+1
#   h = box[2]-box[0]+1
#   area = w*h
#   print(area)

# min_area=[i for i in areas if i>1.25]
# #print(min_area)
# minsize=1.25
# indices = [i for i,v in enumerate(areas) if v >=minsize]
# print(len(areas))
# print(indices)

def get_min_boxes(result, areas, minsize=1.25):
  indexes = [i for i,v in enumerate(areas) if v >=minsize]
  min_boxes = []
  for index in indexes:
    min_boxes.append(result['detection_boxes'][index])
  return min_boxes

# get_min_boxes = get_min_boxes(new_result)

# print(len(get_min_boxes))
# for box in get_min_boxes:
#   w = box[3]-box[1]+1
#   h = box[2]-box[0]+1
#   area = w*h
#   print(area)

#  draw = ImageDraw.Draw(image)
#   im_width, im_height = image.size
#   (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
#                                 ymin * im_height, ymax * im_height)
# print(downloaded_image_path.split('.')[0].split('/')[2])

#from tensorflow.keras.preprocessing.image import load_img, img_to_array

# model = keras.models.load_model('/content/drive/My Drive/Flowers insight project/model2.h5')
# labels = ['daisy', 'ranunculus', 'roses', 'sunflowers', 'tulips']

def predictions(image, model, threshold=.1):
  labels = ['calla lily', 'dahlia', 'daisy', 'iris', 'lily', 'peony', 'ranunculus', 'rose', 'sunflower', 'tulip']
  preds = model.predict_proba(image, verbose=1)
  #print(preds)
  img_label = []
  for i in range(0, len(preds[[0]][0])):
    if preds[[0]][0][i]>threshold:
      img_label.append(labels[i])
  return img_label

#directory = "/content/drive/My Drive/Flowers insight project/crops"

def unique(list1): 
    x = np.array(list1) 
    return(np.unique(x))
    

def crop_box(top_boxes, downloaded_image_path, directory, model):
  predsdict = {}
  #image_array_list = []
  # im=Image.open(downloaded_image_path)
  # image_array=load_img(downloaded_image_path).numpy()
  # image=Image.fromarray(image_array).convert("RGB")
  # im_width, im_height = image.size
  filedir=downloaded_image_path.split('.')[0].split('/')[-1]
  for box in range(0, len(top_boxes)):
    image_array=load_img(downloaded_image_path).numpy()
    image=Image.fromarray(image_array).convert("RGB")
    im_width, im_height = image.size
    bbox = tuple(top_boxes[box])
    #print(bbox)
    
    ymin = bbox[0]
    xmin = bbox[1]
    ymax = bbox[2]
    xmax = bbox[3]
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
    a,b,c,d = int(left) , int(right) , int(top) ,int(bottom)

    image_array = image_array[c:d,a:b]
    converted=tf.image.convert_image_dtype((cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)), tf.float32)[tf.newaxis, ...]
    #print(converted.shape[2])
    full_filename = directory+"/{}_crop{}.jpg".format(filedir, box)
    cv2.imwrite(full_filename, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    filename = 'crops/'+full_filename.split('/')[-1]
    try:
      preds = predictions(converted, model = model, threshold=.3)
      #print(box, preds)
    except:
      preds = ['could not make classification']
      pass
    #image_array_list.append(preds)
    #flattened = [val for sublist in image_array_list for val in sublist]
    predsdict[filename]=preds
  return predsdict
 #unique(flattened)
#preds = crop_box(get_min_boxes, downloaded_image_path, directory)  

#print(image_array)
#display_image(Image.fromarray(image_array))
#image=Image.fromarray(image_array)
#print(image.size)
#image_array2=image.crop((left, right, top, bottom))
#print(image_array2.size)
#display_image(image_array2)

# unique = unique(preds)
# print(unique)

# for i in image_array_list:
#   try:
#     preds = predictions(i, model = model)
#     print(preds)
#   except:
#     pass

