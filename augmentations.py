import pandas as pd
import numpy as np
import copy, cv2
from label_convertors import *

class BaseAugmentation:

    def __init__(self):
        self.total_aug = 0

    def augment(self, img, annotations):
        return img, annotations


class NoiseAugmentation(BaseAugmentation):

    def __init__(self, mean, var):
        super().__init__()
        self.mean = mean
        self.var = var
        self.sigma = var ** 0.5

    def augment(self, img, annotations):
        H, W, C = img.shape
        gauss = np.random.normal(self.mean, self.sigma, (H, W, C))
        gauss = gauss.reshape(H, W, C)
        noise_img = img + gauss
        return noise_img, annotations


class HsvAugmentation(BaseAugmentation):

    def __init__(self, h_r, s_r, v_r):
        super(HsvAugmentation, self).__init__()
        self.h_r = h_r
        self.s_r = s_r
        self.v_r = v_r

    def augment(self, img, annotations):
        h_r = np.random.uniform(1 - self.h_r, 1 + self.h_r)
        s_r = np.random.uniform(1 - self.s_r, 1 + self.s_r)
        v_r = np.random.uniform(1 - self.v_r, 1 + self.v_r)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv_img, dtype=np.float64)
        hsv[:, :, 0] = hsv[:, :, 0] * h_r
        hsv[:, :, 0][hsv[:, :, 0] > 255] = 255
        hsv[:, :, 1] = hsv[:, :, 1] * s_r
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2] * v_r
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        hsv_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return hsv_img, annotations


class GaussBlurAugmentation(BaseAugmentation):

    def __init__(self, max_k):
        super(GaussBlurAugmentation, self).__init__()
        self.max_k = max_k

    def augment(self, img, annotations):
        k = np.random.randint(1, self.max_k)
        gauss_img = cv2.GaussianBlur(img, (k, k), cv2.BORDER_DEFAULT)
        return gauss_img, annotations


class AddObjectsAugmentation(BaseAugmentation):

    def __init__(self, max_list, max_objects):
        super(AddObjectsAugmentation, self).__init__()
        self.max_list = max_list
        self.max_objects = max_objects
        self.queue = []

    def add_to_queue(self, img, object):
        self.total_aug += 1
        if (len(self.queue) == self.max_list):
            self.queue.__delitem__(np.random.randint(self.max_list))
        annotation = convert_corn_label_box(object)

        points = self.get_points_from_object(object)

        x, y, w, h = annotation['min_x'], annotation['min_y'], annotation['max_x']-annotation['min_x'], annotation['max_y']-annotation['min_y']
        croped = img[y:y + h, x:x + w].copy()
        pts = points - points.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.fillPoly(mask, pts=np.array([pts], dtype=np.int32), color=255)
        #cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        cv2.imwrite(f'{self.total_aug}.jpg', dst)
        annotation['max_y'] -= annotation['min_y']
        annotation['max_x'] -= annotation['min_x']
        annotation['min_x'] = 0
        annotation['min_y'] = 0
        self.queue.append([dst, annotation, pts])

    def get_points_from_object(self, object):
        polygon = object['polygon']
        df = pd.DataFrame(polygon)
        return df.to_numpy()

    def augment(self, img, annotations):
        number_new_objs = np.random.randint(0, self.max_objects)
        new_img = img.copy()
        new_annotations = copy.deepcopy(annotations)
        for i in range(number_new_objs):
        #for i in range(number_new_objs):
            new_img, new_annotations = self.augment_img(new_img, new_annotations)
        return new_img, new_annotations

    def augment_img(self, img, annotations):
        add_img, add_annotation, pts = self.queue[np.random.randint(len(self.queue))]
        H, W, C = img.shape
        img2 = img.copy()
        p_x, p_y = np.random.randint(W-add_annotation['max_y']+add_annotation['min_y']),\
                   np.random.randint(H-add_annotation['max_x']+add_annotation['min_x'])
        new_annotation = self.create_new_annotation(add_annotation, p_x, p_y)
        cv2.fillPoly(img2, pts=np.array([pts + np.array([p_x, p_y])], dtype=np.int32), color=(0, 0, 0))
        self.copy_img(img2, add_img, new_annotation)
        annotations.append(new_annotation)
        return img2, annotations

    def create_new_annotation(self, add_annotation, px, py):
        new_annotation = copy.deepcopy(add_annotation)
        new_annotation['min_y'] += py
        new_annotation['max_y'] += py
        new_annotation['max_x'] += px
        new_annotation['min_x'] += px
        return new_annotation

    def copy_img(self, croped_img, add_img, new_annotation):
        min_x, max_x, min_y, max_y = new_annotation['min_x'], new_annotation['max_x'], \
                                     new_annotation['min_y'], new_annotation['max_y']

        croped_img[min_y:max_y, min_x:max_x, :] += add_img

