import pandas as pd
import numpy as np
import copy, cv2
from label_convertors import *
import albumentations as a
import logging

class BaseAugmentation:

    def __init__(self):
        self.total_aug = 0

    def augment(self, img, annotations):
        return img, annotations

    def register(self, img, object):
        pass



class AlbumentationWrapper(BaseAugmentation):

    def __init__(self, transforms, name="AlbumentationWrapper"):
        super(AlbumentationWrapper, self).__init__()
        self.transform = transforms
        self.name = name

    def augment(self, img, annotations):
        logging.debug(f'[{self.name}::Augment] input annotations = {annotations}')
        bboxes = self.convert_to_albumentations_format(annotations)
        logging.debug(f'[{self.name}::Augment] converted bboxes = {bboxes}')
        transformed = self.transform(image=img, bboxes=bboxes)
        logging.debug(f'[{self.name}::Augment] Transformed Annotations = {transformed["bboxes"]}')
        ret_annotations = self.convert_to_annotations_format(transformed['bboxes'])
        logging.debug(f'[{self.name}::Augment] Transformed Annotations : {ret_annotations}')
        return transformed['image'], ret_annotations


    @staticmethod
    def convert_to_annotations_format(bboxes):
        logging.debug("[AlbumentationWrapper::convert_to_augmentations_format")
        annotations = []
        for bbox in bboxes:
            annotation = {}
            annotation['min_x'], annotation['min_y'], annotation['max_x'], annotation['max_y'] = \
                bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]
            annotations.append(annotation)
        return annotations

    @staticmethod
    def convert_to_albumentations_format(annotations):
        bboxes = []
        for annotation in annotations:
            x, y, w, h = annotation['min_x'], annotation['min_y'], annotation['max_x']-annotation['min_x'], annotation['max_y']-annotation['min_y']
            bboxes = [int(x), int(y), int(w), int(h)]
        return bboxes


class NoiseAugmentation(BaseAugmentation):

    def __init__(self, mean, var):
        super().__init__()
        self.mean = mean
        self.var = var
        self.sigma = var ** 0.5

    def augment(self, img, annotations):
        H, W, C = img.shape
        gauss = np.random.normal(self.mean, self.sigma, (H, W, C))
        gauss = 255*gauss.reshape(H, W, C)
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


class HlsAugmentation(BaseAugmentation):

    def __init__(self, h_r, l_r, s_r):
        super(HlsAugmentation, self).__init__()
        self.h_r = h_r
        self.s_r = s_r
        self.l_r = l_r

    def augment(self, img, annotations):
        h_r = np.random.uniform(1 - self.h_r, 1 + self.h_r)
        s_r = np.random.uniform(1 - self.s_r, 1 + self.s_r)
        l_r = np.random.uniform(1 - self.l_r, 1 + self.l_r)
        hsl_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsl = np.array(hsl_img, dtype=np.float64)
        hsl[:, :, 0] = hsl[:, :, 0] * h_r
        hsl[:, :, 0][hsl[:, :, 0] > 255] = 255
        hsl[:, :, 2] = hsl[:, :, 1] * s_r
        hsl[:, :, 2][hsl[:, :, 1] > 255] = 255
        hsl[:, :, 1] = hsl[:, :, 2] * l_r
        hsl[:, :, 1][hsl[:, :, 2] > 255] = 255
        hsl = np.array(hsl, dtype=np.uint8)
        hsl_img = cv2.cvtColor(hsl, cv2.COLOR_HLS2BGR)
        return hsl_img, annotations


class GaussBlurAugmentation(BaseAugmentation):

    def __init__(self, max_k):
        super(GaussBlurAugmentation, self).__init__()
        self.max_k = max_k

    def augment(self, img, annotations):
        k = np.random.randint(1, self.max_k)
        k += k%2 == 0 #Just to make sure the random number is odd there could be a better way of doing it
        gauss_img = cv2.GaussianBlur(img, (k, k), cv2.BORDER_DEFAULT)
        return gauss_img, annotations


class AddObjectsAugmentation(BaseAugmentation):

    def __init__(self, max_list, max_objects):
        super(AddObjectsAugmentation, self).__init__()
        self.max_list = max_list
        self.max_objects = max_objects
        self.queue = []

    def register(self, img, object, debug=False):
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
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        if debug:
            cv2.imwrite(f'{self.total_aug}.jpg', dst)
        self.queue.append([dst, pts])

    def get_points_from_object(self, object):
        polygon = object['polygon']
        df = pd.DataFrame(polygon)
        return df.to_numpy()

    def augment(self, img, annotations):
        number_new_objs = np.random.randint(0, self.max_objects)
        new_img = img.copy()
        new_annotations = copy.deepcopy(annotations)
        for i in range(number_new_objs):
            new_img, new_annotations = self.augment_img(new_img, new_annotations)
        return new_img, new_annotations

    def augment_img(self, img, annotations):
        object, pts = self.get_random_object()
        H, W, C = img.shape
        object_height, object_width, _ = object.shape
        p_x, p_y = np.random.randint(W - object_width),\
                   np.random.randint(H - object_height)
        img2, new_annotation = self.add_object(img, pts, object, p_x, p_y)
        annotations.append(new_annotation)
        return img2, annotations

    def get_random_object(self):
        obj_intrst = np.random.randint(len(self.queue))
        return self.queue[obj_intrst]

    def add_object(self, img, pts, object, px, py):
        img2 = img.copy()
        h, w, _ = img2.shape
        object_height, object_width, _ = object.shape
        new_annotation = self.create_new_annotation(object_width, object_height, px, py, h, w)
        cv2.fillPoly(img2, pts=np.array([pts + np.array([px, py])], dtype=np.int32), color=(0, 0, 0))
        self.copy_img(img2, object, new_annotation)
        return img2, new_annotation


    def create_new_annotation(self, object_width, object_height, px, py, img_h, img_w):
        new_annotation = {}
        new_annotation['min_y']  = py
        new_annotation['max_y']  = min(object_height + py, img_h)
        new_annotation['max_x']  = min(px + object_width, img_w)
        new_annotation['min_x']  = px
        return new_annotation

    def copy_img(self, croped_img, add_img, new_annotation):
        min_x, max_x, min_y, max_y = new_annotation['min_x'], new_annotation['max_x'], \
                                     new_annotation['min_y'], new_annotation['max_y']

        croped_img[min_y:max_y, min_x:max_x, :] += add_img[:max_y-min_y, :max_x-min_x]


class DoubleSeedAugmentation(AddObjectsAugmentation):

    def __init__(self, q_size, prob=0.1):
        super(DoubleSeedAugmentation, self).__init__(q_size, 1)
        self.prob = prob #prob of creating a double

    def augment(self, img, annotations):
        new_annotations = copy.copy(annotations)
        new_img = img.copy()

        if(np.random.rand()<self.prob):
            new_img, new_annotations = self.apply_double_aug(new_img, new_annotations)

        return new_img, new_annotations

    def apply_double_aug(self, new_img, new_annotation):
        object, pts = self.get_random_object()
        px, py = self.get_random_point(new_annotation)
        new_img, annotation = self.add_object(new_img, pts, object, px, py)
        #cv2.imwrite('Test.jpg', new_img)
        new_annotation.append(annotation)
        return new_img, new_annotation

    def get_random_point(self, new_annotation):
        ann = np.random.choice(new_annotation)
        h, w = ann['max_y'] - ann['min_y'], ann['max_x'] - ann['min_x']
        r_h, r_w = np.random.random(), np.random.random()

        if(r_h > 0.5):
            py = ann['max_y'] - (1-r_h)*h
        else:
            py = ann['min_y'] + (r_h)*h

        if(r_w > 0.5):
            px = ann['max_x'] - (1-r_w)*w
        else:
            px = ann['min_x'] + (r_w)*w

        return int(px), int(py)