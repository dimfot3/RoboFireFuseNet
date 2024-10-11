import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './datasets/')
from base_dataset import BaseDataset


class WildFire(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes=2,
                 multi_scale=True, 
                 flip=True, 
                 brightness=True,
                 contrast=True,
                 single_source=False,
                 ignore_label=255, 
                 base_size=1024, 
                 crop_size=(720, 960),
                 scale_factor=16,
                 mean=[0, 0, 0, 0],
                 std=[1, 1, 1, 1],
                 bd_dilate_size=4, 
                 mode='rgb'):

        indices = {'rgb': [0, 1, 2], 'ir': [3], 'fusion': [0, 1, 2, 3]}      
        self.mean = [mean[i] for i in indices[mode]]
        self.std = [std[i] for i in indices[mode]]
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

        self.single_source = single_source

        self.mode = mode

    
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

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        if self.mode=='rgb':
            image = np.array(Image.open(os.path.join(self.root,item["img"])).convert('RGB'))
        elif self.mode=='ir':
            image = np.array(Image.open(os.path.join(self.root,item["ir"])).convert('L'))
            image = image.reshape(image.shape[0], image.shape[1], 1)
        elif self.mode=='fusion':
            image = np.array(Image.open(os.path.join(self.root,item["ir"])).convert('L'))
            image = image.reshape(image.shape[0], image.shape[1], 1)
            image1 = np.array(Image.open(os.path.join(self.root,item["img"])).convert('RGB'))
            image = np.append(image1, image, 2)
        size = image.shape
        color_map = Image.open(os.path.join(self.root,item["label"])).convert('RGB')
        color_map = np.array(color_map)
        label = self.color2label(color_map)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, brightness=self.brightness, contrast=self.contrast)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

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
    dataset = WildFire('./data/CorsicanDB/', 'list/train.txt')
    mean = np.array(dataset.mean)
    std = np.array(dataset.std)
    for i in np.random.choice(len(dataset), 3):
        img, label, edge, size, _ = dataset[i]
        f, ax = plt.subplots(1, 3)
        img = img.transpose((1, 2, 0))
        img = (((img * std) + mean) * 255).astype('int')
        gt_img = dataset.label2color(label).astype('int')
        edge = edge.astype('int')
        edge[edge==1] = 255
        ax[0].imshow(img)
        ax[1].imshow(edge)
        ax[2].imshow(gt_img)
        plt.show()

        
