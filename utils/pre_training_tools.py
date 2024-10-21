import torch
import numpy as np
import torch.nn.functional as F
import os
from .tools import get_confusion_matrix
from .total_loss import MaskedMSELoss
from .scheduler import CustomPolynomialDecayLR
import torch.optim as optim
from models.pidnet import PIDNet
from models.AsyncModel import AsyncModel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datasets.imagenet import ImageNet


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
        images, labels, mask = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), batch[2].to(dtype=torch.long, device=self.device)
        output = self.model(images)
        output_mask = F.interpolate(
                            output[1],
                            size=[images.shape[-2], images.shape[-1]],
                            mode='bilinear', align_corners=True)
        loss = self.criterion.get_loss(output_mask, labels, mask)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach()

    def valid_step(self, batch):
        self.model.eval()
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
            optimizer = optim.AdamW(model.parameters(), lr=args['LR'], betas=(0.9, 0.95))
        else:
            print('Unsupported optimizer.')
            exit()
        return optimizer

    def get_scheduler(self, initial_lr, max_iters):
        scheduler = CustomPolynomialDecayLR(self.optimizer, initial_lr, max_iters=max_iters)
        return scheduler

    def get_loss_criterion(self, args):
        loss = MaskedMSELoss(args)
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
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = ImageNet(args['ROOTDATASET'], transform_train)
    return dataset

def get_model(args):
    if 'pidnet_s' == args['MODEL']:
        model = PIDNet(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=32, ppm_planes=96, head_planes=128, augment=True, channels=12)
    elif 'pidnet_m' == args['MODEL']:
        model = PIDNet(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=96, head_planes=128, augment=True, channels=12)
    elif 'pidnet_l' == args['MODEL']:
        model = PIDNet(m=3, n=4, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=112, head_planes=256, augment=True, channels=12)
    elif 'async_s' == args['MODEL']:
        model = AsyncModel(64)
    elif 'async_m' == args['MODEL']:
        model = AsyncModel(128)
    if args['PRETRAINED'] is not None:
        model.load_state_dict(torch.load(args['PRETRAINED'], map_location='cpu'))
    model.to(device=args['DEVICE'])
    return model
