import torch
import numpy as np
import torch.nn.functional as F
import os
from .tools import get_confusion_matrix
from .total_loss import MaskedMSELoss
from .scheduler import CosineDecay
import torch.optim as optim
from models.pidnet import PIDNet
from models.AsyncModel import AsyncModel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datasets.imagenet import ImageNet


class Trainer:
    def __init__(self, args, model, len_data):
        self.use_amp = False if args['DEVICE'] == 'cpu' else True
        self.model = model
        self.optimizer = self.get_optimizer(args, self.model)
        self.criterion = self.get_loss_criterion(args)
        self.scheduler = self.get_scheduler(args['LR'], args['EPOCHS'], np.ceil(len_data / args['BATCHSIZE']), args['WARMUP'])
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.device = args['DEVICE']
        self.num_classes = args['NUM_CLASSES']
        self.ingore_label = args['IGNORE_LABEL']
        self.update_freq = args['UPDATE_FREQ']
        self.iter_counter = 1
        self.start_epoch = 0
        if args['CHECKPOINT'] != None:
            self.load_checkpoint(os.path.join(os.path.join('weights', args['PROJECTNAME'], args['SESSIONAME'], args['CHECKPOINT'])))
           
    def training_step(self, batch):
        self.model.train()
        images, labels, mask = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), batch[2].to(dtype=torch.long, device=self.device)
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
            output = self.model(images)
            output_mask = F.interpolate(
                                output[1],
                                size=[images.shape[-2], images.shape[-1]],
                                mode='bilinear', align_corners=True)
            loss = self.criterion.get_loss(output_mask, labels, mask)
        loss = loss.mean() / self.update_freq
        self.scaler.scale(loss).backward()
        if (self.iter_counter % self.update_freq) == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.iter_counter = 0
            self.optimizer.zero_grad()
        self.iter_counter += 1
        return loss.detach() * self.update_freq

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
        # output = torch.argmax(output, dim=1)
        return output

    def get_optimizer(self, args, model):
        optimizer = optim.AdamW(model.parameters(), lr=args['LR'], betas=(0.9, 0.95))
        return optimizer

    def get_scheduler(self, initial_lr, epochs, num_batches, warmup):
        scheduler = CosineDecay(self.optimizer, initial_lr, epochs, num_batches, warmup)
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
    
    def save_checkpoint(self, path, epoch, itter=0):
        os.makedirs(f'{path}', exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }
        torch.save(checkpoint, os.path.join(path, f'checkpoint_epoch_{epoch}_{itter}.pth'))

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch']

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
