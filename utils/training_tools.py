import torch
import numpy as np
import torch.nn.functional as F
import os
from .tools import get_confusion_matrix, get_confusion_matrix_instancewise
from .total_loss import TotalLoss
from .scheduler import CosineDecay, PolynomialDecayLR
import torch.optim as optim
from models.pidnet import PIDNet
from datasets.wildfire import WildFire
from models.Asyncv2 import PIDnetTF, make_square_input
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF


class Trainer:
    def __init__(self, args, model, len_data, test=False):
        self.model_name = args['MODEL']
        self.use_amp = False if args['DEVICE'] == 'cpu' else True
        self.model = model
        self.optimizer = self.get_optimizer(args, self.model)
        self.criterion = self.get_loss_criterion(args)
        self.scheduler = self.get_scheduler(args['SCHED'], args['LR'], args['EPOCHS'], (np.ceil(len_data / args['BATCHSIZE'])), args['WARMUP'])
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.device = args['DEVICE']
        self.num_classes = args['NUM_CLASSES']
        self.ingore_label = args['IGNORE_LABEL']
        self.stop_counter = args['STOPCOUNTER']
        self.start_epoch = 0
        self.best_metric = 0
        self.stop_cur_counter = 0
        if args['CHECKPOINT'] != None:
            self.load_checkpoint(os.path.join(os.path.join('weights', args['PROJECTNAME'], args['SESSIONAME'], args['CHECKPOINT'])), test)

    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        images, labels, edges, names = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), batch[2].to(dtype=torch.float, device=self.device), batch[3]
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
            output = self.model(images)
            output_mask = F.interpolate(
                                output[1],
                                size=[images.shape[-2], images.shape[-1]],
                                mode='bilinear', align_corners=True)
            losses, _, acc, loss_list = self.criterion.get_loss(output, labels, edges)
        conf_mat = get_confusion_matrix(labels, output_mask, self.num_classes, ignore=self.ingore_label)
        loss = losses.mean()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss.detach(), conf_mat

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

    
    def test_step(self, batch):
        self.model.eval()
        images, labels, edges, names = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), batch[2].to(dtype=torch.float, device=self.device), batch[3]
        output = self.model(images)
        output_mask = F.interpolate(
                            output[1],
                            size=[images.shape[-2], images.shape[-1]],
                            mode='bilinear', align_corners=True)
        conf_mat = get_confusion_matrix(labels, output_mask, self.num_classes, ignore=self.ingore_label)
        conf_mat_inst = get_confusion_matrix_instancewise(labels, output_mask, self.num_classes, ignore=self.ingore_label, k=0.1)
        losses, _, acc, loss_list = self.criterion.get_loss(output, labels, edges)
        loss = losses.mean()
        return loss.detach(), conf_mat, conf_mat_inst

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

    def get_scheduler(self, sched_name, initial_lr, epochs, num_batches, warmup):
        if sched_name == 'COS':
            scheduler = CosineDecay(self.optimizer, initial_lr, epochs, num_batches, warmup)
        elif sched_name == 'POLY':
            scheduler = PolynomialDecayLR(self.optimizer, initial_lr, epochs * num_batches, num_batches, 0.9, 10, warmup_epochs=warmup)
        return scheduler

    def get_loss_criterion(self, args):
        loss = TotalLoss(args)
        return loss

    def stop_sign(self, metrics):
        if metrics['val miou'] - self.best_metric > 0.01:
            self.best_metric, self.stop_cur_counter = metrics['val miou'], 0
        else:
            self.stop_cur_counter += 1
        if self.stop_cur_counter >= self.stop_counter:
            return True
        return False
    
    def save_checkpoint(self, path, epoch):
        os.makedirs(f'{path}', exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_metric': self.best_metric
        }
        torch.save(checkpoint, os.path.join(path, f'checkpoint_epoch_{epoch}.pth'))

    def load_checkpoint(self, path, test=False):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if(not test):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.start_epoch = checkpoint['epoch'] - 1
        self.best_metric = checkpoint['best_metric']
        print('Checkpoint loaded!')

def get_dataset(args, test=False):
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
                          n_stack=args['N_FRAMES'],
                          frames_appart=args['MAX_FR_APART'],
                          mode=args['MODE'])
    
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
                          n_stack=args['N_FRAMES'],
                          frames_appart=args['MAX_FR_APART'],
                          mode=args['MODE'])
    if test:
        test_dataset = WildFire(root=args['ROOTDATASET'],
                          list_path=args['TESTSET'],
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
                          n_stack=args['N_FRAMES'],
                          frames_appart=args['MAX_FR_APART'],
                          mode=args['MODE'])
        return train_dataset, val_dataset, test_dataset
    return train_dataset, val_dataset

def get_model(args):
    if 'pidnet_s' == args['MODEL']:
        model = PIDNet(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=32, ppm_planes=96, head_planes=128, augment=True, channels=3 if ('MODE' != 'fusion') else 4)
    elif 'pidnet_m' == args['MODEL']:
        model = PIDNet(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=96, head_planes=128, augment=True, channels=3 if ('MODE' != 'fusion') else 4)
    elif 'pidnet_l' == args['MODEL']:
        model = PIDNet(m=3, n=4, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=112, head_planes=256, augment=True, channels=3 if ('MODE' != 'fusion') else 4)
    elif 'async_s' == args['MODEL']:
        model = PIDnetTF(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=32, ppm_planes=96, head_planes=128, augment=True, channels=4, layer5='tf', input_resolution=args['BASE_SIZE'], window_size=int(args['BASE_SIZE']//64))
    elif 'async_m' == args['MODEL']:
        model = PIDnetTF(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=96, head_planes=128, augment=True, channels=4, layer5='conv', input_resolution=args['BASE_SIZE'], window_size=int(args['BASE_SIZE']//64))
    if args['PRETRAINED'] is not None:
        model.imgnet_pretrain(args['PRETRAINED'])
    model.to(device=args['DEVICE'])
    return model