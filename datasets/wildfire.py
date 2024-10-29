import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './datasets/')
from base_dataset import BaseDataset
import re
import pandas as pd


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
                 frames_appart=5,
                 seed=200,
                 load_cache=True):

        self.mean = mean
        self.std = std
        super(WildFire, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, self.mean, self.std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.load_cache = load_cache
        self.multi_scale = multi_scale
        self.flip = flip
        self.brightness = brightness
        self.contrast = contrast
        self.img_list = [line[:-1] for line in open(os.path.join(root, list_path))]
        self.files = self.read_files()
        self.ignore_label = ignore_label
        # self.color_list = [[0, 0, 0], [125, 125, 125],[255, 255, 255]]
        self.color_list = [
        (0, 0, 0),          # 0:    background(unlabeled)
        (0, 0, 142),        # 1:    Car
        (0, 60, 100),       # 2:    Bus
        (0, 0, 230),        # 3:    Motorcycle
        (119, 11, 32),      # 4:    Bicycle
        (255, 0, 0),        # 5:    Pedestrian
        (0, 139, 139),      # 6:    Motorcyclist
        (255, 165, 150),    # 7:    Bicyclist
        (192, 64, 0),       # 8:    Cart
        (211, 211, 211),    # 9:    Bench
        (100, 33, 128),     # 10:   Umbrella
        (117, 79, 86),      # 11:   Box
        (153, 153, 153),    # 12:   Pole
        (190, 122, 222),    # 13:   Street_lamp
        (250, 170, 30),     # 14:   Traffic_light
        (220, 220, 0),      # 15:   Traffic_sign
        (222, 142, 35),     # 16:   Car_stop
        (205, 155, 155),    # 17:   Color_cone
        (70, 130, 180),     # 18:   Sky
        (128, 64, 128),     # 19:   Road
        (244, 35, 232),     # 20:   Sidewalk
        (0, 0, 70),         # 21:   Curb
        (107, 142, 35),     # 22:   Vegetation
        (152, 251, 152),    # 23:   Terrain
        (70, 70, 70),       # 24:   Building
        (110, 80, 100),      # 25:   Ground
        (255, 255, 255)      # 26:   ignore
        ]

        self.class_weights = None
        self.bd_dilate_size = bd_dilate_size
        self.n_stack = n_stack
        self.frames_appart = frames_appart
        self.seed = seed

    
    def read_files(self):
        cache_file_path = os.path.join(self.root, '.cache_df.csv')
        if os.path.exists(cache_file_path) and self.load_cache:
            print("Loading from cache...")
            return pd.read_csv(cache_file_path)
        
        labeled_files = []
        for item in self.img_list:
            folder, item = '/'.join(item.split('/')[:-1]), item.split('/')[-1]
            name = os.path.splitext(os.path.basename(item))[0]
            id_finder = re.search(r'^(.*?)(?:_I?(\d{5})_|[(](\d+)[)])', name)
            labeled_files.append({
                "folder": folder,
                "name": name,
                "prefix": id_finder.group(1),
                "id": int(id_finder.group(2) or id_finder.group(3)),
                "labeled": True})
        df = pd.DataFrame(labeled_files)
        unique_folders = df['folder'].unique()
        for folder in unique_folders:
            folder_path = os.path.join(self.root, folder)
            all_files = [file for file in os.listdir(folder_path) if (file.find('_rgb') != -1)]
            for file_name in all_files:
                file_name = file_name.replace('_rgb_', '_XXX_').replace('_rgb', '_XXX')
                name = os.path.splitext(file_name)[0]
                id_finder = re.search(r'^(.*?)(?:_I?(\d{5})_|[(](\d+)[)])', name)
                prefix = id_finder.group(1)
                id_value = int(id_finder.group(2) or id_finder.group(3))
                if not (df['name'] == name).any():
                    labeled_files.append({
                        "folder": folder,
                        "name": name,
                        "prefix": prefix,
                        "id": id_value,
                        "labeled": False  # Mark as not labeled
                    })
        df = pd.DataFrame(labeled_files)
        df.to_csv(os.path.join(self.root, '.cache_df.csv'), index=False)
        print("DataFrame saved to cache.")
        return df

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
    
    def filter_by_prefix_and_id_range(self, df, row_index, k):
        target_row = df.iloc[row_index]
        target_prefix = target_row['prefix']
        target_id = int(target_row['id'])
        same_prefix_df = df[df['prefix'] == target_prefix]
        filtered_df = same_prefix_df[(same_prefix_df['id'].astype(int) >= target_id - k) &
                                    (same_prefix_df['id'].astype(int) <= target_id + k)]
        return target_row, filtered_df[filtered_df['name'] != target_row['name']]

    def pick_target_with_candidates(self, candidates, target_row):
        target_name = target_row['name']
        folder = target_row['folder']
        numbers = np.sort(candidates['id'].to_numpy())
        interval = len(numbers) / (self.n_stack - 2) if self.n_stack > 2 else 0
        indices = [round(i * interval) for i in range(self.n_stack - 1)]
        selected_numbers = [numbers[i] for i in indices if i < len(numbers)]
        selected_candidates = candidates[candidates['id'].isin(selected_numbers)]
        images = [os.path.join(folder, name) for name in selected_candidates['name'].tolist()]
        images.append(os.path.join(folder, target_name))
        names = selected_candidates['name'].tolist()
        names.append(target_name)
        return images, names
    
    def get_async_multi_modal_inputs(self, list_of_imgs):
        modal_file_paths = []
        start_with = np.random.choice(['rgb', 'ir'])
        for i in range(len(list_of_imgs)):
            replacement = start_with if i % 2 == 0 else 'ir' if start_with == 'rgb' else 'rgb'
            modal_file_paths.append(os.path.join(self.root, list_of_imgs[i].replace('XXX', replacement)))
        label_path = os.path.join(self.root, list_of_imgs[-1].replace('XXX', 'gt') + '.png')        # label is in png as there must be no compression
        extensions = ['png', 'jpg']
        existing_files = []
        for path in modal_file_paths:
            directory = os.path.dirname(path)
            file_name = os.path.basename(path)
            for ext in extensions:
                full_path = os.path.join(directory, file_name + '.' + ext)
                if os.path.exists(full_path):
                    existing_files.append(full_path)
        return existing_files, label_path

    def __getitem__(self, index):
        target, cands = self.filter_by_prefix_and_id_range(self.files, index, self.frames_appart)
        list_of_imgs, names = self.pick_target_with_candidates(cands, target)
        images, label = self.get_async_multi_modal_inputs(list_of_imgs)
        loaded_images = []
        for path in images:
            with Image.open(path).convert('L' if '_ir' in path else 'RGB') as img:
                img = np.asarray(img)
                img = img.reshape(*(img.shape[:2]), -1)
                loaded_images.append(img)
        with Image.open(label) as label_img:
            label = np.asarray(label_img)
            label = self.label2color(label) if len(label.shape) == 2 else label
        images, label, edge = self.gen_sample(loaded_images, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, brightness=self.brightness, contrast=self.contrast)
        for i, img in enumerate(images):
            if img.shape[0] == 1:
                images[i] = np.append(images[i], np.zeros((2, images[i].shape[1], images[i].shape[2])), axis=0)
        images = np.concatenate(images, axis=0)
        return images, label, edge, names

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
    dataset = WildFire(root='../Datasets/',
                          list_path='lists/mvseg_train.txt',
                          num_classes=3,
                          multi_scale=True,
                          flip=True,
                          brightness=True,
                          contrast=True,
                          ignore_label=26,
                          scale_factor=9,
                          crop_size=[272, 336],
                          base_size=336,
                          bd_dilate_size=4,
                          n_stack=2,
                          frames_appart=10,
                          load_cache=True)
    for i in np.random.choice(len(dataset), 3):
        images, label, edge, name = dataset[i]
        images = images.reshape(len(images) // 3, 3, images.shape[-2], images.shape[-1])
        f, ax = plt.subplots(1, len(images) + 2)
        for i, img in enumerate(images):
            img = (img * 255).astype('int')
            img = np.transpose(img, (1, 2, 0))
            if img[:, :, 1:].sum() == 0:
                img = img[:, :, 0]
            ax[i].imshow(img)
        label = label.astype('int')
        edge = edge.astype('int')
        ax[-2].imshow(label)
        ax[-1].imshow(edge)
        plt.show()
        
        

        
