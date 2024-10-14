import torch
import numpy as np
import torch.nn.functional as F
import os
from .tools import get_confusion_matrix
from .total_loss import TotalLoss
from .scheduler import CustomPolynomialDecayLR
import torch.optim as optim
from models.pidnet import PIDNet
from datasets.wildfire import WildFire


class Trainer:
    def __init__(self, args, model, len_data):
        self.model = model
        self.optimizer = self.get_optimizer(args, self.model)
        self.criterion = self.get_loss_criterion(args)
        self.scheduler = self.get_scheduler(args['LR'], (np.ceil(len_data / args['BATCHSIZE'])) * args['EPOCHS'])
        self.device = args['DEVICE']
        self.num_classes = args['NUM_CLASSES']
        self.ingore_label = args['IGNORE_LABEL']

    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        images, labels, edges, names = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), batch[2].to(dtype=torch.float, device=self.device), batch[3]
        output = self.model(images)
        output_mask = F.interpolate(
                            output[1],
                            size=[images.shape[-2], images.shape[-1]],
                            mode='bilinear', align_corners=True)
        conf_mat = get_confusion_matrix(labels, output_mask, self.num_classes, ignore=self.ingore_label)
        losses, _, acc, loss_list = self.criterion.get_loss(output, labels, edges)
        loss = losses.mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach(), conf_mat

    def valid_step(self, batch):
        self.model.eval()
        images, labels, edges, names = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), batch[2].to(dtype=torch.float, device=self.device), batch[4]
        output = self.model(images)
        output_mask = F.interpolate(
                            output[1],
                            size=[images.shape[-2], images.shape[-1]],
                            mode='bilinear', align_corners=True)
        conf_mat = get_confusion_matrix(labels, output_mask, self.num_classes, ignore=self.ingore_label)
        losses, _, acc, loss_list = self.criterion.get_loss(output, labels, edges)
        loss = losses.mean()
        return loss.detach(), conf_mat

    def inference(self, data):
        self.model.eval()
        images = data.to(dtype=torch.float, device=self.device)
        output = self.model(images)
        output = F.interpolate(
                            output[1],
                            size=[images.shape[-2], images.shape[-1]],
                            mode='bilinear', align_corners=True)
        output = torch.argmax(output, dim=1)
        return output

    def get_optimizer(self, args, model):
        if args['OPTIM'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args['LR'], momentum=args['MOMENTUM'], weight_decay=args['WD']) 
        elif args['OPTIM'] == 'ADAM':
            optimizer = optim.Adam(model.parameters(), lr=args['LR'], weight_decay=args['WD'])
        else:
            print('Unsupported optimizer.')
            exit()
        return optimizer

    def get_scheduler(self, initial_lr, max_iters):
        scheduler = CustomPolynomialDecayLR(self.optimizer, initial_lr, max_iters=max_iters)
        return scheduler

    def get_loss_criterion(self, args):
        loss = TotalLoss(args)
        return loss

    def stop_sign(self, metrics):
        return False
        if metrics['val avg_f1'] > self.best_metric:
            self.best_metric, self.counter = metrics['val avg_f1'], 0
        else:
            self.counter += 1
        if self.counter >= self.args['STOPCOUNTER']:
            return True
        return False

def get_dataset(args):
    train_dataset = WildFire(root=args['ROOTDATASET'],
                          list_path=args['TRAINSET'],
                          num_classes=args['NUM_CLASSES'],
                          multi_scale=args['MULTISCALE'],
                          flip=args['FLIP'],
                          brightness=args['BRIGHTNESS'],
                          contrast=args['CONTRAST'],
                          ignore_label=args['IGNORE_LABEL'],
                          scale_factor=args['SCALE_FACTOR'],
                          crop_size=args['CROP_SIZE'],
                          base_size=args['BASE_SIZE'],
                          bd_dilate_size=4,
                          n_stack=3,
                          frames_appart=args['MAX_FR_APART']) 
    val_dataset = WildFire(root=args['ROOTDATASET'],
                          list_path=args['VALIDSET'],
                          num_classes=args['NUM_CLASSES'],
                          multi_scale=False,
                          flip=False,
                          brightness=False,
                          contrast=False,
                          ignore_label=args['IGNORE_LABEL'],
                          scale_factor=args['SCALE_FACTOR'],
                          crop_size=args['CROP_SIZE'],
                          base_size=args['BASE_SIZE'],
                          bd_dilate_size=4,
                          n_stack=3,
                          frames_appart=args['MAX_FR_APART'])
    return train_dataset, val_dataset

def get_model(args):
    if 'pidnet_s' == args['MODEL']:
        model = PIDNet(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=32, ppm_planes=96, head_planes=128, augment=True, channels=12)
    elif 'pidnet_m' == args['MODEL']:
        model = PIDNet(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=96, head_planes=128, augment=True, channels=12)
    elif 'pidnet_l' == args['MODEL']:
        model = PIDNet(m=3, n=4, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=112, head_planes=256, augment=True, channels=12)
    if args['PRETRAINED'] is not None:
        model.load_state_dict(torch.load(args['PRETRAINED'], map_location='cpu'))
    model.to(device=args['DEVICE'])
    return model