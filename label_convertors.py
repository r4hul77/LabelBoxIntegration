import pandas as pd
from label_convertors import *

def convert_corn_label_box(corn_object):
    polygon = corn_object['polygon']
    df = pd.DataFrame(polygon)
    return {'label': 0, 'min_x': int(min(df['x'])), 'max_x' : int(max(df['x']))+1, 'min_y' : int(min(df['y'])), 'max_y' : int(max(df['y']))+1}

def convert_blurry_label_box(blurry_object):
    bbox = blurry_object['bbox']
    return {'label' : 0, 'min_x': bbox['left'], 'max_x' : bbox['left'] + bbox['width'], 'min_y' : bbox['top'], 'max_y' : bbox['top'] + bbox['height']}
