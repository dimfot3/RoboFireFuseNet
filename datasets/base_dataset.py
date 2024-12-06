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
                 mean=[0, 0, 0, 0],
                 std=[1, 1, 1, 1]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean_rgb = mean[:3]
        self.mean_ir = mean[3]
        self.std_rgb = std[:3]
        self.std_ir = std[3]
        self.scale_factor = scale_factor

    def input_transform(self, images):
        """
        Read the image and do the following steps:
            2) normalize the images by scaling to 0,1 and removing mean and divide with std
        """
        for i, image in enumerate(images):
            image = image.astype(np.float32)
            image = image / 255.0
            image = image - (self.mean_rgb if image.shape[-1] == 3 else self.mean_ir)
            image = image / (self.std_rgb if image.shape[-1] == 3 else self.std_ir)
            images[i] = image
        return images

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
        if len(image.shape) == 3:
            n_ch = image.shape[-1]
            pad_image = np.resize(pad_image, (pad_image.shape[0], pad_image.shape[1], n_ch))
        return pad_image

    def rand_crop(self, images, label, edge):
        """
        Random crop images in dimension where there are less than self.crop_size
        """
        h, w = images[0].shape[:-1]
        for i, image in enumerate(images):
            images[i] = self.pad_image(image, h, w, self.crop_size,
                                (0.0, 0.0, 0.0))
        
        label = self.pad_image(label, h, w, self.crop_size,
                               (255.0, 255.0, 255.0))
        edge = self.pad_image(edge, h, w, self.crop_size,
                               (0.0,))
        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        for i, image in enumerate(images):
            images[i] = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        edge = edge[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        return images, label, edge

    def multi_scale_aug(self, images, label=None, edge=None,
                        rand_scale=1, rand_crop=True):
        """
        Randomly changes the scale of an image
        """
        long_size = int(self.base_size * rand_scale + 0.5)
        h, w = images[0].shape[:2]
        if h > w:
            new_h = long_size
            new_w = int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = int(h * long_size / w + 0.5)
        for i, image in enumerate(images):
            nc = images[i].shape[-1]
            images[i] = cv2.resize(image, (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR).reshape(new_h, new_w, nc)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                            interpolation=cv2.INTER_NEAREST)
            if edge is not None:
                edge = cv2.resize(edge, (new_w, new_h),
                                interpolation=cv2.INTER_NEAREST).reshape(new_h, new_w)
        else:
            return images
        if rand_crop:
            images, label, edge = self.rand_crop(images, label, edge)
        return images, label, edge

    def day_to_night(self, image):
        gamma = 1 + 1.5 * np.random.rand()
        look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        darkened = cv2.LUT(image, look_up_table)
        
        tint = np.zeros_like(darkened, dtype=np.float32)
        tint[..., 0] += 20
        tinted = cv2.addWeighted(darkened.astype(np.float32), 1.0, tint, 0.1, 0)

        rows, cols = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols / 2)
        kernel_y = cv2.getGaussianKernel(rows, rows / 2)
        mask = kernel_y * kernel_x.T
        mask = (mask / mask.max())
        vignette = (tinted * mask[..., np.newaxis]).astype(np.uint8)

        return vignette

    def night_to_day(self, image):
        # Brighten the image using gamma correction
        gamma = 1 - 0.5 * np.random.rand()
        look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        brightened = cv2.LUT(image, look_up_table)

        # Enhance color saturation
        hsv = cv2.cvtColor(brightened, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.add(hsv[..., 1], 50)  # Increase saturation
        saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Add a warm (yellowish) tint
        tint = np.zeros_like(saturated, dtype=np.float32)
        tint[..., 1] += 20  # Add green
        tint[..., 2] += 40  # Add red
        warmed = cv2.addWeighted(saturated.astype(np.float32), 1.0, tint, 0.1, 0)

        # Adjust contrast slightly for a daytime look
        alpha = 1.2  # Contrast control
        beta = 20    # Brightness control
        adjusted = cv2.convertScaleAbs(warmed, alpha=alpha, beta=beta)

        return adjusted

    def is_night_based_on_brightness(self, image, threshold=80):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness < threshold

    def change_brightness(self, images):
        for i, img in enumerate(images):
            if (img.shape[-1] == 1) or (img[:, :, 1:].sum() == 0):
                continue
            else:
                images[i] = self.night_to_day(img) if self.is_night_based_on_brightness(img) else self.day_to_night(img)
        return images

    def complementary_masking(self, image1, image2, patch_size=32):
        H, W, _ = image1.shape
        patch_mask = np.random.choice([0, 1], size=(H // patch_size + 1, W // patch_size + 1))
        masked_image1 = np.zeros_like(image1)
        masked_image2 = np.zeros_like(image2)
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                h_end = min(i + patch_size, H)
                w_end = min(j + patch_size, W)
                
                if patch_mask[i // patch_size, j // patch_size] == 1:
                    masked_image1[i:h_end, j:w_end, :] = image1[i:h_end, j:w_end, :]
                else:
                    masked_image2[i:h_end, j:w_end, :] = image2[i:h_end, j:w_end, :]
        return masked_image1, masked_image2

    def gen_sample(self, images, label,
                   multi_scale=True, is_flip=True, edge_pad=True, edge_size=4, brightness=True, comp_mask=False, single_source=False):
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
        if brightness and (np.random.random() > 0.5):
            images = self.change_brightness(images)
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            images, label, edge = self.multi_scale_aug(images, label, edge,
                                                rand_scale=rand_scale)
        else:
            images, label, edge = self.multi_scale_aug(images, label, edge,
                                                rand_scale=1, rand_crop=True)
        images = self.input_transform(images)
        label = self.label_transform(label)

        if comp_mask and (np.random.rand() < 0.1):
            images[0], images[1] = self.complementary_masking(images[0], images[1], 32)
        if single_source and (np.random.rand() < 0.1):
            p = np.random.randint(0, 2)
            images[0], images[1] = (1 - p) * images[0], images[1]* p

        images = [image.transpose(2, 0, 1) for image in images]

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            label = label[:, ::flip]
            edge = edge[:, ::flip]
            for i, image in enumerate(images):
                images[i] = image[:, :, ::flip]
        return images, label, edge


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

