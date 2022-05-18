# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import labelbox
import urllib
import json
import requests
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tqdm
from datetime import date
#Seed
np.random.seed(69)
# Enter your Labelbox API key here
LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDJ0NzF6aHQzYzV4MDc5M2dwbGE2eXZrIiwib3JnYW5pemF0aW9uSWQiOiJjbDJ0NzF6aGgzYzV3MDc5M2hteGcxc2pvIiwiYXBpS2V5SWQiOiJjbDNhaGs1c3ExNDFoMDc3M2RtcG9oYmRuIiwic2VjcmV0IjoiYTAyYzhmNDIwMjIyNDFmZWQ5YTg5MWIxMGY5YWVjYWIiLCJpYXQiOjE2NTI4MTE5NDgsImV4cCI6MjI4Mzk2Mzk0OH0.Tsp0Fxrzesx9D7RRHh3PVv9JJwqAFk-jegdsZ_7zp_Q"

# Create Labelbox client

lb = labelbox.Client(api_key=LB_API_KEY)

# Get project by ID

project = lb.get_project('cl2t7enc9ehdb076f78hrg8n0')

# Export image and text data as an annotation generator:


with urllib.request.urlopen(project.export_labels()) as url:
    labels_export = json.loads(url.read().decode())

splited_labels = {}
splited_labels['train'], splited_labels['test'] = train_test_split(labels_export, test_size=0.2)
splited_labels['train'], splited_labels['val']  = train_test_split(splited_labels['train'], test_size=0.2)

def mkdir(dir, name):
    named_dir = os.path.join(os.path.join(dir, name))
    if(not os.path.isdir(named_dir)):
        os.mkdir(named_dir)
        os.mkdir(os.path.join(named_dir, 'images'))
        os.mkdir(os.path.join(named_dir, 'labels'))
    return named_dir

def setup_dir_for_yolo(dir):
    dirs = ['train', 'val', 'test']
    ret = {}
    for named_dir in dirs:
        ret[named_dir]=mkdir(dir,named_dir)
    create_yaml_file(dir, ret)
    return ret

def get_quality_imgs(labels_export):
    total_seeds = 0
    total_good_data = 0
    total_corn_seeds = 0
    total_blurry_seeds = 0
    for label in labels_export:
        if(len(label['Label']['objects']) > 0):
            if(label['Label']['objects'][0]['title'] != 'no_seed'):
                total_good_data += 1
                total_seeds += len(label['Label']['objects'])
                for object in label['Label']['objects']:
                    if(object['title']=='corn_seed'):
                        total_corn_seeds += 1
                    else:
                        total_blurry_seeds += 1
    return {'total_seeds': total_seeds, 'total_good_imgs': total_good_data, 'total_corn_seeds':total_corn_seeds, 'total_blurry':total_blurry_seeds}
# Press the green button in the gutter to run the script.

def download_save_img(label, img_name):
    response = requests.get(label['Labeled Data'])
    file = open(img_name, 'wb')
    file.write(response.content)
    file.close()

def convert_corn_label_box(corn_object):
    polygon = corn_object['polygon']
    df = pd.DataFrame(polygon)
    return {'label': 0, 'min_x': min(df['x']), 'max_x' : max(df['x']), 'min_y' : min(df['y']), 'max_y' : max(df['y'])}

def convert_blurry_label_box(blurry_object):
    bbox = blurry_object['bbox']
    return {'label' : 0, 'min_x': bbox['left'], 'max_x' : bbox['left'] + bbox['width'], 'min_y' : bbox['top'], 'max_y' : bbox['top'] + bbox['height']}

def convert_yolo_format(annotation, H=175., W=480.):
    x_c = (annotation['min_x'] + annotation['max_x'])/(2*W)
    y_c = (annotation['min_y'] + annotation['min_y'])/(2*H)
    h   = (annotation['max_y'] - annotation['min_y'])/H
    w   = (annotation['max_x'] - annotation['min_x'])/W
    return {'label' : 0, 'x_c' : x_c, 'y_c' : y_c, 'h' : h, 'w' : w}

def save_labels_in_yolo_format(labels, dir):
    count_good_imgs = -1
    pbar = tqdm.tqdm(labels)
    pbar.set_description(dir.split('/')[-1])
    for label in pbar:
        annotations = []
        if(len(label['Label']['objects']) > 0):
            if(label['Label']['objects'][0]['title'] != 'no_seed'):
                count_good_imgs += 1
                for object in label['Label']['objects']:
                    if(object['title'] =='corn_seed'):
                        temp_ann = convert_corn_label_box(object)
                    elif(object['title']=='blury_seed'):
                        temp_ann = convert_blurry_label_box(object)
                    else:
                        continue
                    annotations.append(convert_yolo_format(temp_ann))
                if(len(annotations) > 0):
                    download_save_img(label, f'{dir}/images/{count_good_imgs}.JPEG')
                    save_annotations_yolo_format(annotations, f'{dir}/labels/{count_good_imgs}.txt')

def save_annotations_yolo_format(annotations, txt_file):
    with open(txt_file, 'w') as file:
        for annotation in annotations:
            file.write(' '.join(map(lambda  x : str(x), annotation.values()))+'\n')

def create_yaml_file(dir, dirs):
    with open(os.path.join(dir, 'data.yaml'), 'w') as f:
        print(dirs)
        for named_dir in dirs.keys():
            f.write(f"{named_dir} : {dirs[named_dir]}/\n")
        f.write("\n")
        f.write("nc : 1\nnames : ['seed']")

if __name__ == '__main__':
    dir = os.path.join(os.path.expanduser("~"), date.today().strftime("%m-%d"), 'yolov5')
    os.makedirs(dir, exist_ok=True)
    dirs = setup_dir_for_yolo(dir)
    for dir in dirs.keys():
        save_labels_in_yolo_format(splited_labels[dir], dirs[dir])



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
