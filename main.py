# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#!/home/r4hul/PycharmProjects/LabelBox/venv/bin/python

import requests, io, cv2, tqdm, urllib, labelbox, json, os, copy
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from datetime import date
import logging
from augmentations import *
from label_convertors import *
logging.basicConfig(filename='main.log', level=logging.DEBUG)
#Seed
np.random.seed(69)
# Enter your Labelbox API key here
LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDJ0NzF6aHQzYzV4MDc5M2dwbGE2eXZrIiwib3JnYW5pemF0aW9uSWQiOiJjbDJ0NzF6aGgzYzV3MDc5M2hteGcxc2pvIiwiYXBpS2V5SWQiOiJjbDNhaGs1c3ExNDFoMDc3M2RtcG9oYmRuIiwic2VjcmV0IjoiYTAyYzhmNDIwMjIyNDFmZWQ5YTg5MWIxMGY5YWVjYWIiLCJpYXQiOjE2NTI4MTE5NDgsImV4cCI6MjI4Mzk2Mzk0OH0.Tsp0Fxrzesx9D7RRHh3PVv9JJwqAFk-jegdsZ_7zp_Q"

# Create Labelbox client

lb = labelbox.Client(api_key=LB_API_KEY)

# Get project by ID

project = lb.get_project('cl2t7enc9ehdb076f78hrg8n0')


#Augmentations

# Export image and text data as an annotation generator:
class BaseTransform:
    #Base Class For Transforms
    def __init__(self):
        self.total_imgs = 0

    def transform(self, img, annotations):
        return img, annotations

class ResizeTransformWidthBased(BaseTransform):

    def __init__(self, target_width):
        super().__init__()
        self.target_width = target_width

    def transform(self, img, annotations):
        logging.debug(f"[ResizeWidthBasedTransform.tranform] input (img shape {img.shape}, annotations {annotations}")
        H, W, _ = img.shape
        r = self.target_width/W
        resize_img = cv2.resize(img, (self.target_width, int(r*H)))
        resized_annotations = copy.deepcopy(annotations)
        for annotation in resized_annotations:
            annotation['min_x'] *= r
            annotation['max_x'] *= r
            annotation['min_y'] *= r
            annotation['max_y'] *= r
        logging.debug(f"[ResizeWidthBasedTransform.tranform] Ouput (img shape {resize_img.shape}, annotations {resized_annotations}")
        return resize_img, resized_annotations

class RandomLetterBoxTransform(BaseTransform):

    def __init__(self, target_shape, fill=[0, 0, 0]): #Target Shape (H, W)
        super().__init__()
        self.target_h = target_shape[0]
        self.target_w = target_shape[1]
        self.fill = fill

    def transform(self, img, annotations):
        logging.debug(f"[RandomLetterBoxTranform.tranform] input (img shape {img.shape}, annotations {annotations})")
        H, W, _ = img.shape
        delta_h = self.target_h - H
        delta_w = self.target_w - W
        t_h = np.random.randint(0, delta_h+1)
        t_w = np.random.randint(0, delta_w+1)
        letterbox_img = cv2.copyMakeBorder(img, t_h, delta_h - t_h, t_w, delta_w - t_w, cv2.BORDER_CONSTANT, value=self.fill)
        lettebox_annotations = copy.deepcopy(annotations)
        for annotation in lettebox_annotations:
            annotation['min_x'] += t_w
            annotation['max_x'] += t_w
            annotation['min_y'] += t_h
            annotation['max_y'] += t_h
        logging.debug(f"[RandomLetterBoxTranform.tranform] input (img shape {letterbox_img.shape}, annotations {lettebox_annotations})")
        return letterbox_img, lettebox_annotations

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

def setup_dir(dir):
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

def download_img(label):
    response = requests.get(label['Labeled Data'])
    bytes_im = io.BytesIO(response.content)
    return cv2.cvtColor(np.array(Image.open(bytes_im)), cv2.COLOR_RGB2BGR)


def convert_yolo_format(annotation, H=175., W=480.):
    x_c = (annotation['min_x'] + annotation['max_x'])/(2*W)
    y_c = (annotation['min_y'] + annotation['min_y'])/(2*H)
    h   = (annotation['max_y'] - annotation['min_y'])/H
    w   = (annotation['max_x'] - annotation['min_x'])/W
    return {'label' : 0, 'x_c' : x_c, 'y_c' : y_c, 'h' : h, 'w' : w}

def save_labels(labels, dir, formats, augment=False, TransformsList=[BaseTransform()], Augmentations_List=[AddObjectsAugmentation(10, 4)]):
    yolo_format = False
    coco_format = False
    coco_dict = None
    for format in formats:
        if(format=='yolov5'):
            yolo_format = True
        if(format=="coco"):
            coco_format = True
            coco_dict = create_coco_dict()
    total_imgs = 0
    augmented_imgs  = 0

    pbar = tqdm.tqdm(labels)
    pbar.set_description(dir.split('/')[-1])
    annotation_number     = 0
    augmented_annotations = 0
    for label in pbar:
        yolo_annotations = []
        coco_annotations = []
        annotations = []
        if(len(label['Label']['objects']) > 0):
            if(label['Label']['objects'][0]['title'] != 'no_seed'):
                img = download_img(label)
                apply_augmentation = False
                for object in label['Label']['objects']:
                    if(object['title'] =='corn_seed'):
                        apply_augmentation = True*augment
                        if(apply_augmentation):
                            Augmentations_List[0].add_to_queue(img, object)
                        temp_ann = convert_corn_label_box(object)
                    elif(object['title']=='blury_seed'):
                        temp_ann = convert_blurry_label_box(object)
                    else:
                        continue
                    annotations.append(temp_ann)
                if(len(annotations) > 0):


                    for transform in TransformsList:
                        img, annotations = transform.transform(img, annotations)

                    if apply_augmentation:
                        imgs, annotations_list = apply_augmentations(img, annotations, Augmentations_List)
                        augmented_annotations += len(annotations_list)
                        augmented_imgs += len(imgs)
                    else:
                        imgs = [img]
                        annotations_list = [annotations]
                    for img, annotations in zip(imgs, annotations_list):
                        for annotation in annotations:
                            yolo_annotations.append(convert_yolo_format(annotation))
                            coco_annotations.append(convert_coco_format(annotation, annotation_number + augmented_annotations,
                                                                        total_imgs))
                            annotation_number += 1
                        save_data_row(img, yolo_format, coco_format, yolo_annotations, coco_annotations, total_imgs, coco_dict, dir)
#                        if(yolo_format):
#                            save_annotations_yolo_format(yolo_annotations, f'{dir}/labels/{count_good_imgs}.txt')
#                        if(coco_format):
#                            coco_dict = save_annotations_coco_format(coco_annotations, coco_dict, count_good_imgs)
                        total_imgs += 1

    if coco_format:
        with open(f'{dir}/images/_annotations.coco.json', 'w') as f:
            json.dump(coco_dict, f, indent=4)
    print(f"Wrote {total_imgs} Images with {annotation_number} annotations into {dir} with {augmented_imgs} augmented Imgs and {augmented_annotations} augmented annotations")

def save_data_row(img, yolo_format, coco_format, yolo_annotations, coco_annotations, img_id, coco_dict, dir):
    cv2.imwrite(f'{dir}/images/{img_id}.jpg', img)
    H, W, C = img.shape
    if (yolo_format):
        save_annotations_yolo_format(yolo_annotations, f'{dir}/labels/{img_id}.txt')
    if (coco_format):
        coco_dict = save_annotations_coco_format(coco_annotations, coco_dict, img_id, H, W)
    return coco_dict

def convert_coco_format(annoation, annotation_number, img_id):
    w = int(annoation['max_x'] - annoation['min_x'])
    h = int(annoation['max_y'] - annoation['min_y'])
    return {'id': annotation_number,
            'image_id': img_id,
            'category_id': 0,
            'bbox': [int(annoation['min_x']),
                     int(annoation['min_y']),
                     w,
                     h],
            'area': h*w,
            'segmentation': [],
            'iscrowd': 0}



def save_annotations_coco_format(annotations, dict, img_id, H, W):
    dict['images'].append({'id': img_id,
                           'license': 1,
                           'file_name': f'{img_id}.jpg',
                           'height': H,
                           'width': W,
                           'date_captured': date.today().strftime("%Y-%m-%d")})

    dict['annotations'].append(annotations)

    return dict

def create_coco_dict():
    return {'info': {'year': date.today().strftime("%Y"),
                     'version': '0',
                     'description': 'Created By the Script',
                     'contributor': 'r4hul',
                     'url': '',
                     'date_created': date.today().strftime("%Y-%m-%d")},

            'licenses': [{'id': 1,
                          'url': 'https://creativecommons.org/licenses/by/4.0/',
                          'name': 'CC BY 4.0'}],

            'categories': [{'id': 0, 'name': 'seed', 'supercategory': 'none'}],
                           #,{'id': 1, 'name': 'seed', 'supercategory': 'seeds'}]

            'images' : [],

            'annotations' : []

            }







def apply_augmentations(img, annotation, list_of_augmentations=[BaseAugmentation(), BaseAugmentation(), BaseAugmentation()]):
    ret_imgs, ret_annotations = [], []
    for augmentation in list_of_augmentations:
        aug_img, aug_annotation = augmentation.augment(img, annotation)
        ret_imgs.append(aug_img)
        ret_annotations.append(aug_annotation)
    return ret_imgs, ret_annotations


def augment_coco(coco, idx, img_idx):
    ret = copy.deepcopy(coco)
    for i, ann in enumerate(ret) :
        ret[i]["id"] += idx*len(coco) + i
        ret[i]["image_id"] = idx + img_idx
    return ret

def save_annotations_yolo_format(annotations, txt_file):
    with open(txt_file, 'w') as file:
        for annotation in annotations:
            file.write(' '.join(map(lambda  x : str(x), annotation.values()))+'\n')

def create_yaml_file(dir, dirs):
    with open(os.path.join(dir, 'data.yaml'), 'w') as f:
        for named_dir in dirs.keys():
            f.write(f"{named_dir} : {dirs[named_dir]}/\n")
        f.write("\n")
        f.write("nc : 1\nnames : ['seed']")

if __name__ == '__main__':
    dir = os.path.join(os.path.expanduser("~"), "dataset_test", date.today().strftime("%m-%d"))
    os.makedirs(dir, exist_ok=True)
    dirs = setup_dir(dir)
    for dir in dirs.keys():
        save_labels(splited_labels[dir][:100], dirs[dir], ["yolov5", "coco"], augment=dir == "train", TransformsList=[ResizeTransformWidthBased(512), RandomLetterBoxTransform((512, 512))])
    print("Wrote Imgs and Labels Smoothly")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
