# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import cv2
import numpy as np
import random

from torch.nn import functional as F
from torch.utils import data

y_k_size = 6
x_k_size = 6

class BaseDataset(data.Dataset):
    def __init__(self,
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0, 0, 0],
                 std=[1, 1, 1]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        """
        Read the image and do the following steps:
            2) normalize the images by scaling to 0,1 and removing mean and divide with std
        """
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        """
        It transform the labels in numpy array of int8
        """
        return np.array(label).astype(np.uint8)

    def pad_image(self, image, h, w, size, padvalue):
        """
        This pads the dimension of the image that is less than
        a predifined size filling with a padvalue.
        """
        pad_image = image.copy()
        pad_h = max(int(size[0]) - h, 0)
        pad_w = max(int(size[1]) - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)
        return pad_image

    def rand_crop(self, image, label, edge):
        """
        Random crop images in dimension where there are less than self.crop_size
        """
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))
        edge = self.pad_image(edge, h, w, self.crop_size,
                               (0.0,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        edge = edge[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        if(len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)
        return image, label, edge

    def multi_scale_aug(self, image, label=None, edge=None,
                        rand_scale=1, rand_crop=True):
        """
        Randomly changes the scale of an image
        """
        long_size = int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if(len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
            if edge is not None:
                edge = cv2.resize(edge, (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            return image
        if rand_crop:
            image, label, edge = self.rand_crop(image, label, edge)
        return image, label, edge

    def change_brightness(self, image):
        brightness_factor = 0.5 + np.random.rand(1)
        float_image = image.astype(np.float32)
        brightened_image = float_image * brightness_factor
        brightened_image = np.clip(brightened_image, 0, 255).astype(np.uint8)
        return brightened_image

    def adjust_contrast(self, image):
        float_image = image.astype(np.float32)
        mean = np.mean(float_image, axis=(0, 1), keepdims=True)
        contrast_image = (float_image - mean) * (np.random.random((1, )) + 0.5) + mean
        contrast_image = np.clip(contrast_image, 0, 255).astype(np.uint8)
        return contrast_image
    
    def hide_one_source(self, image):
        source = np.random.randint(0,2,1)
        idxs = [[0,1,2], [3]]
        image[idxs[source]] = 0
        return image

    def gen_sample(self, image, label,
                   multi_scale=True, is_flip=True, edge_pad=True, edge_size=4, brightness=True, contrast=True, single_source=False):
        """
        generate a training sample by applying augmentation, then generates edge label with cv2 and then 
        normalizes the images and the label and return image, label and edges
        """
        edge = cv2.Canny(label, 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        if edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0
        
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label, edge = self.multi_scale_aug(image, label, edge,
                                                rand_scale=rand_scale)            
        if brightness and (np.random.random() > 0.85):
            image = self.change_brightness(image)
        if contrast and (np.random.random() > 0.85):
            image = self.adjust_contrast(image)
        if single_source and (np.random.random() > 0.75):
            image = self.adjust_contrast(image)
        image = self.input_transform(image)
        label = self.label_transform(label)
        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            edge = edge[:, ::flip]

        return image, label, edge


    def inference(self, config, model, image):
        """
        Pass the image through the model and return the expodential results
        """
        size = image.size()
        _, pred, _ = model(image)
        
        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config['ALIGN_CORNERS']
        )
        return pred.exp()

