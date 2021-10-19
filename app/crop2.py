#crop the bounding boxes into individual images

import json
import cv2
import numpy as np
import csv
import os

with open('result.json') as f:  #Enter file path for JSON file
  data = json.load(f)   #list

for idx, sales in enumerate(data):
    os.chdir(r'/home/user/src/darknet')
    dict1 = data[idx]   #dict
    objlist = dict1['objects']  #list
    bbox = []
    for item in objlist:
      if item['name'] in ('price' , 'date'):
        dict2 = item['relative_coordinates']
        x1 = dict2['center_x']
        y1 = dict2['center_y']
        w1 = dict2['width']
        h1 = dict2['height']
        name = item['name']
        list1 = [x1 , y1 , w1 , h1 , name]
        bbox.append(list1)

    img = cv2.imread(r'/home/user/src/darknet/' + dict1['filename'])  #<--- Parameter: Enter file path for Image
    fname = dict1['filename']
    fn = fname.replace('salebox_','')
    fn = fn.replace('.jpg','')
    fn = fn.replace('proj/eb/saleboxes/','')
    print(fn)
    
    for idxx, bbox in enumerate(bbox):
      x2 = bbox[0]
      y2 = bbox[1]
      w2 = bbox[2]
      h2 = bbox[3]
      height, width, channels = img.shape
      x = int(x2 * width)
      y = int(y2 * height)
      w = int(w2 * width)
      h = int(h2 * height)
      crop_img = img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
      if bbox[4] == 'date':
        os.chdir(r'/home/user/src/darknet/proj/eb/textboxes/dates')
        cv2.imwrite("tb_{}_date.jpg".format(fn), crop_img)
      else:
        os.chdir(r'/home/user/src/darknet/proj/eb/textboxes/prices')
        cv2.imwrite("tb_{}_price.jpg".format(fn), crop_img)