import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './datasets/')
from base_dataset import BaseDataset
import re


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
                 n_stack=5,
                 frames_appart=210):

        self.mean = mean
        self.std = std
        super(WildFire, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, self.mean, self.std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        self.brightness = brightness
        self.contrast = contrast
        self.img_list = [line[:-1] for line in open(os.path.join(root, list_path))]
        self.files = self.read_files()
        self.ignore_label = ignore_label
        self.color_list = [[0, 0, 0], [125, 125, 125],[255, 255, 255]]
        self.class_weights = None
        self.bd_dilate_size = bd_dilate_size
        self.n_stack = n_stack
        self.frames_appart = frames_appart

    
    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, ir_path, label_path = item.replace('XXX', 'rgb'), item.replace('XXX', 'ir'), item.replace('XXX', 'gt')
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "ir": ir_path,
                "label": label_path,
                "name": name.replace('_gt_', '_').replace('_gt', '')
            })
        return files
        
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
    
    def find_closest_images(self, target_id, k):
        bounds = [[0, 8100], [8100, 9000], [9000, 100000]] # TODO fill the bounds
        idx = next((i for i, (low, high) in enumerate(bounds) if low <= target_id < high), None)
        lower_bound = max(bounds[idx][0], target_id - k)
        ids = np.arange(max(1, target_id - k), target_id)
        cand_ids = np.append(np.random.choice([image_id for image_id in ids if lower_bound <= image_id < target_id], \
                                              min(self.n_stack, len(ids)), replace=False), target_id).astype('int')
        cand_ids.sort()
        modes = ['rgb', 'ir']
        closest_filenames = [f'img_{modes[np.random.randint(0, 2)]}_({image_id}).png' for image_id in cand_ids]
        return closest_filenames

    def __getitem__(self, index):
        item = self.files[index]
        name, target_id = item["name"], int(re.search(r'img_\((\d+)\)', item["name"]).group(1))
        img_folder = os.path.join(self.root, '/'.join(item['img'].split('/')[:2]))
        images = self.find_closest_images(target_id, self.frames_appart)
        color_map = Image.open(os.path.join(self.root,item["label"])).convert('RGB')
        color_map = np.array(color_map)
        label = self.color2label(color_map)
        images = [np.asarray(Image.open(os.path.join(img_folder, image)).convert('RGB'))
                   if 'rgb' in image else np.asarray(Image.open(os.path.join(img_folder, image)).convert('L'))[:, :, np.newaxis] for image in images]
        images, label, edge = self.gen_sample(images, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, brightness=self.brightness, contrast=self.contrast)
        for i, img in enumerate(images):
            if img.shape[0] == 1:
                images[i] = np.append(images[i], np.zeros((2, images[i].shape[1], images[i].shape[2])), axis=0)
        images = np.concatenate(images, axis=0)
        return images, label.copy(), edge.copy(), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

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
            image = (((images[i] * self.std) + self.mean) * 255).astype('int')
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
    dataset = WildFire(root='../../Datasets/',
                          list_path='lists/train_flm.txt',
                          num_classes=3,
                          multi_scale=True,
                          flip=True,
                          brightness=True,
                          contrast=True,
                          ignore_label=255,
                          scale_factor=9,
                          crop_size=[272, 336],
                          base_size=336,
                          bd_dilate_size=4,
                          n_stack=4,
                          frames_appart=210)
    for i in np.random.choice(len(dataset), 3):
        images, label, edge, name = dataset[i]
        f, ax = plt.subplots(1, len(images))
        print(name)
        for i, img in enumerate(images):
            img = (img * 255).astype('int')
            img = np.transpose(img, (1, 2, 0))
            ax[i].imshow(img)
        plt.show()
        

        
