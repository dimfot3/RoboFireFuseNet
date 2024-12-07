import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './datasets/')
from base_dataset import BaseDataset
import re
import pandas as pd
import torch
import torchvision

class WildFire(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes=2,
                 multi_scale=True, 
                 flip=True, 
                 brightness=True,
                 contrast=True,
                 ignore_label=255, 
                 base_size=1024, 
                 crop_size=(720, 960),
                 scale_factor=16,
                 mean=[0, 0, 0, 0],
                 std=[1, 1, 1, 1],
                 bd_dilate_size=4, 
                 seed=200,
                 mode='fusion',
                 blend_images=False,
                 robust_train=False,
                 comp_mask=False,
                 single_source=False):

        self.mean = mean
        self.std = std
        super(WildFire, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, self.mean, self.std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.mode = mode
        self.multi_scale = multi_scale
        self.flip = flip
        self.brightness = brightness
        self.contrast = contrast
        self.files = [line for line in open(os.path.join(root, list_path)).read().split('\n') if len(line) > 0]
        self.ignore_label = ignore_label
        self.robust_train = robust_train
        self.color_list = [[0, 0, 0], [125, 125, 125],[255, 255, 255]] if num_classes == 3 else [
        (0, 0, 0),          # 0:    background(unlabeled)
        (64, 0, 128),        # 1:    Car
        (64, 64, 0),       # 2:    person
        (0, 128, 192),        # 3:    bike
        (0, 0, 192),      # 4:    curve
        (128, 128, 0),        # 5:    car_stop
        (64, 64, 128),      # 6:    guardrail
        (192, 128, 128),    # 7:    color_cone
        (192, 64, 0)       # 8:    bump
        ]
        self.bd_dilate_size = bd_dilate_size
        self.seed = seed
        self.blend_images = blend_images
        self.comp_mask = comp_mask
        self.single_source = single_source

    def __len__(self):
        return len(self.files)

    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2])*self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2)==3] = i
        return label.astype(np.uint8)
    
    def label2color(self, label):
        color_map = np.zeros(label.shape+(3,))
        for i, v in enumerate(self.color_list):
            color_map[label==i] = self.color_list[i]
        return color_map.astype(np.uint8)
    
    def random_transformation_matrix(self, scale_range=(0.9, 1.1), 
                                 rotation_range=(-8, 8), 
                                 translation_range=(-15, 15)):
       
        # Random scaling
        scale_x = np.random.uniform(*scale_range)
        scale_y = scale_x

        # Random rotation
        theta = np.radians(np.random.uniform(*rotation_range))
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        # Random translation
        trans_x = np.random.uniform(*translation_range)
        trans_y = np.random.uniform(*translation_range)

        # Create the transformation matrix
        transformation_matrix = np.array([
            [scale_x * cos_theta, -scale_y * sin_theta, trans_x],
            [scale_x * sin_theta,  scale_y * cos_theta, trans_y],
            [0,                   0,                   1]
        ])
        
        return transformation_matrix, scale_x, (trans_x, trans_y), theta

    def blend_objects(self, img1_list, mask1, img2_list, mask2):
        hor_shift, ver_shift = np.random.randint(0, 500), np.random.randint(0, 500)
        mask1 = np.roll(np.roll(mask1, hor_shift, axis=1), ver_shift, axis=0)
        for i, img1 in enumerate(img1_list):
            img1_list[i] = np.roll(np.roll(img1, hor_shift, axis=1), ver_shift, axis=0)
        obj_ids = np.unique(mask1)
        obj_ids = obj_ids[obj_ids != 0]
        if len(obj_ids) == 0:   return img2_list, mask2
        obj_id = np.random.choice(obj_ids, 1)
        obj_mask = (mask1 == obj_id)
        mask2[obj_mask] = mask1[obj_mask]
        for i, img2 in enumerate(img2_list):
            img2_list[i][obj_mask] = img1_list[i][obj_mask]
        return img2_list, mask2

    def load_sample(self, index):
        rgb_img = np.asarray(Image.open(os.path.join(self.root, self.files[index].replace('XXX', 'rgb'))).convert('RGB')).copy()
        ir_img = np.asarray(Image.open(os.path.join(self.root, self.files[index].replace('XXX', 'ir'))).convert('L')).copy()
        ir_img = ir_img.reshape(*(ir_img.shape[:2]), -1)
        label_img = np.asarray(Image.open(os.path.join(self.root, self.files[index].replace('XXX', 'gt'))).convert('RGB')).astype('uint8').copy()
        unique_labels = np.unique(label_img)
        unique_labels = unique_labels[unique_labels!=self.ignore_label]
        label_img = label_img[:, :, 0] if unique_labels.max() < 100 else self.color2label(label_img) # dangarous line
        loaded_images = [rgb_img, ir_img]
        return loaded_images, label_img

    def __getitem__(self, index):
        loaded_images, label = self.load_sample(index)
        # blend augmentation
        if self.blend_images and (np.random.rand() < 0.5):
            for i in range(3):
                tmp_loaded_images, tmp_label = self.load_sample(np.random.randint(0, len(self)))
                loaded_images, label = self.blend_objects(tmp_loaded_images, tmp_label, loaded_images, label)
        # add transformation if robust_train
        if self.robust_train:
            _, scale, trans, theta = self.random_transformation_matrix()
            loaded_images[-1], inv_tf = self.transform_image(loaded_images[-1], angle=theta, scale=scale, translate=trans)
        # rest augmentation and image normalizations
        images, label, edge = self.gen_sample(loaded_images, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, brightness=self.brightness, comp_mask=self.comp_mask, single_source=self.single_source)
        images = np.concatenate(images, axis=0)
        # return rgb only, ir only or fusion (default) depending on mode 
        if self.mode == 'rgb':
            images = images[:3]
        elif self.mode == 'ir':
            images = images[3]
        return images.copy(), label.copy(), edge.copy(), [self.files[index].split('/')[-1]], inv_tf if self.robust_train else ()

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def transform_image(self, image, angle=0, scale=1.0, translate=(0, 0)):
        angle = torch.tensor(angle)
        translate = np.array(translate).astype('float')
        translate[0], translate[1] = translate[0] / image.shape[0], translate[1] / image.shape[1]
        scale = scale
        # Scaling matrix
        scaling_matrix = torch.tensor([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ], dtype=torch.float64)
        
        # Rotation matrix
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle),  torch.cos(angle), 0],
            [0, 0, 1]
        ], dtype=torch.float64)
        
        # Translation matrix
        translation_matrix = torch.tensor([
            [1, 0, translate[0]],
            [0, 1, translate[1]],
            [0, 0, 1]
        ], dtype=torch.float64)

        inv_scaling_matrix = torch.tensor([
            [1/scale, 0, 0],
            [0, 1/scale, 0],
            [0, 0, 1]
        ], dtype=torch.float64)
        
        # Rotation matrix
        inv_rotation_matrix = torch.tensor([
            [torch.cos(-angle), -torch.sin(-angle), 0],
            [torch.sin(-angle),  torch.cos(-angle), 0],
            [0, 0, 1]
        ], dtype=torch.float64)
        
        # Translation matrix
        inv_translation_matrix = torch.tensor([
            [1, 0, -translate[0]],
            [0, 1, -translate[1]],
            [0, 0, 1]
        ], dtype=torch.float64)
        
        # Combined transformation: scale -> rotate -> translate
        transformation_matrix =  translation_matrix @ rotation_matrix @ scaling_matrix
        inv_transformation_matrix = inv_scaling_matrix @ inv_rotation_matrix @ inv_translation_matrix
        transformation_matrix = transformation_matrix[:2, :]  # Extract 2x3 affine matrix
        inv_transformation_matrix = inv_transformation_matrix[:2, :].numpy()
        # Create affine grid
        grid = torch.nn.functional.affine_grid(transformation_matrix.unsqueeze(0),
                                                torch.Size([1, image.shape[2], image.shape[0], image.shape[1]]),
                                                align_corners=True)
        input_tensor = torch.from_numpy(image.copy().astype('float64')).permute(2, 0, 1).unsqueeze(0)  # Convert to BCHW

        output_tensor = torch.nn.functional.grid_sample(input_tensor, grid, mode='nearest', padding_mode='zeros', align_corners=True)  
        output_tensor = output_tensor[0].permute(1, 2, 0).numpy().astype(image.dtype)
        return output_tensor, inv_transformation_matrix

    def save_pred(self, images, labels, preds, name, path):
        if(len(preds.shape)>3):
            preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        else:
            preds = np.asarray(preds.cpu(), dtype=np.uint8)
        labels = np.asarray(labels.cpu(), dtype=np.uint8)
        images = np.asarray(images.cpu(), dtype=np.float32).transpose(0, 2, 3, 1)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            label = self.label2color(labels[i])
            if images[i][:, :, 1:3].sum() == 0:
                img = img[:, :, 3:]
            else:
                img = img[:, :, 0:3]
            image = (((img * self.std) + self.mean) * 255).astype('int')
            f, ax = plt.subplots(1, 3, figsize=(10,7))
            titles = ['Image', 'Ground Truth', 'Prediction']
            pics = [image, label, pred]
            for j in range(3):
                ax[j].imshow(pics[j])
                ax[j].set_title(titles[j])
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
            plt.savefig(f'{path}/{name[i]}.png', dpi=300)
            plt.clf()
            plt.close()

        
if __name__ == '__main__':
    dataset = WildFire(root='Datasets/',
                          list_path='lists/train_mfnet.txt',
                          num_classes=3,
                          multi_scale=False,
                          flip=True,
                          brightness=True,
                          contrast=True,
                          ignore_label=26,
                          scale_factor=9,
                          crop_size=[272, 336],
                          base_size=336,
                          bd_dilate_size=4,
                          mode='fusion', robust_train=True, comp_mask=True, single_source=True)
    for idx in np.random.choice(len(dataset), 3):
        images, label, edge, name, tf = dataset[idx]
        f, ax = plt.subplots(1, len(images) + 2)
        images = [images[:3], images[3:]]
        stds, means = [dataset.std_rgb, dataset.std_ir], [dataset.mean_rgb, dataset.mean_ir]
        for i, img in enumerate(images):
            img = np.transpose(img, (1, 2, 0))
            img = (img * stds[i]) + means[i]
            images[i] = (img * 255).astype('uint8')
        label = dataset.label2color(label).astype('uint8')
        edge = edge.astype('uint8')
        plt.imsave(f'rgb_image{idx}.png', images[0])
        plt.imsave(f'ir_image{idx}.png', images[1][:, :, 0])
        plt.imsave(f'label{idx}.png', label)
        plt.imsave(f'edge{idx}.png', edge)
        
