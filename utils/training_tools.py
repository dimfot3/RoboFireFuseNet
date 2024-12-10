import torch
import numpy as np
import torch.nn.functional as F
import os
from .tools import get_confusion_matrix
from .total_loss import TotalLoss
from .scheduler import CosineDecay, PolynomialDecayLR
import torch.optim as optim
from models.pidnet import PIDNet
from datasets.wildfire import WildFire
from models.Asyncv3 import PIDnetTF
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF


class Trainer:
    def __init__(self, args, model, len_data, test=False):
        self.model_name = args['MODEL']
        self.use_amp = False if args['DEVICE'] == 'cpu' else False
        self.model = model
        self.optimizer = self.get_optimizer(args, self.model)
        self.criterion = self.get_loss_criterion(args)
        self.scheduler = self.get_scheduler(args['SCHED'], args['LR'], args['EPOCHS'], (np.ceil(len_data / args['BATCHSIZE'])), args['WARMUP'])
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.device = args['DEVICE']
        self.num_classes = args['NUM_CLASSES']
        self.ingore_label = args['IGNORE_LABEL']
        self.stop_counter = args['STOPCOUNTER']
        self.robust_train = args['ROBUST_TRAIN']
        self.start_epoch = 0
        self.best_metric = 0
        self.stop_cur_counter = 0
        if args['CHECKPOINT'] != None:
            self.load_checkpoint(os.path.join(args['CHECKPOINT']), test)
        if args['ROBUST_TRAIN']:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.robust_module.parameters():
                param.requires_grad = True
            conv_rob = [*(model.conv1_rgb_0_rob.parameters()), *(model.conv1_rgb_0_rob.parameters()), *(model.conv1_rgb_1_rob.parameters()), \
                        *(model.conv1_rgb_2_rob.parameters()), *(model.layer1_rgb_rob.parameters()), *(model.layer2_rgb_rob.parameters()) ]
            for param in conv_rob:
                param.requires_grad = True
            
        # model.robust_module = None

    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        images, labels, edges, names, tf = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), \
        batch[2].to(dtype=torch.float, device=self.device), batch[3], \
        batch[4].to(dtype=torch.float, device=self.device) if self.robust_train else None
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
            output = self.model(images, tf)
            output_mask = F.interpolate(
                                output[1],
                                size=[images.shape[-2], images.shape[-1]],
                                mode='bilinear', align_corners=True)
            losses, _, acc, loss_list = self.criterion.get_loss(output, labels, edges, tf)
        conf_mat = get_confusion_matrix(labels, output_mask, self.num_classes, ignore=self.ingore_label)
        loss = losses.mean()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss.detach(), conf_mat

    def valid_step(self, batch):
        self.model.eval()
        images, labels, edges, names, tf = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), batch[2].to(dtype=torch.float, device=self.device), batch[3], \
        batch[4].to(dtype=torch.float, device=self.device) if self.robust_train else None
        output = self.model(images, tf)
        output_mask = F.interpolate(
                            output[1],
                            size=[images.shape[-2], images.shape[-1]],
                            mode='bilinear', align_corners=True)
        conf_mat = get_confusion_matrix(labels, output_mask, self.num_classes, ignore=self.ingore_label)
        losses, _, acc, loss_list = self.criterion.get_loss(output, labels, edges, tf)
        loss = losses.mean()
        return loss.detach(), conf_mat

    
    def test_step(self, batch):
        self.model.eval()
        images, labels, edges, names, tf = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), batch[2].to(dtype=torch.float, device=self.device), batch[3], \
        batch[4].to(dtype=torch.float, device=self.device) if self.robust_train else None
        output = self.model(images, tf)
        output_mask = F.interpolate(
                            output[1],
                            size=[images.shape[-2], images.shape[-1]],
                            mode='bilinear', align_corners=True)
        conf_mat = get_confusion_matrix(labels, output_mask, self.num_classes, ignore=self.ingore_label)
        losses, _, acc, loss_list = self.criterion.get_loss(output, labels, edges, tf)
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
        elif args['OPTIM'] == 'ADAMW':
            optimizer = optim.AdamW(model.parameters(), lr=args['LR'], weight_decay=args['WD'])
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
        if metrics['val miou'] - self.best_metric > 0.001:
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
            self.start_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        print('Checkpoint loaded!')

def get_dataset(args, test=False):
    train_dataset = WildFire(root=args['ROOTDATASET'],
                          list_path=args['TRAINSET'],
                          num_classes=args['NUM_CLASSES'],
                          multi_scale=args['MULTISCALE'],
                          flip=args['FLIP'],
                          brightness=args['BRIGHTNESS'],
                          ignore_label=args['IGNORE_LABEL'],
                          scale_factor=args['SCALE_FACTOR'],
                          crop_size=args['CROP_SIZE'],
                          base_size=args['BASE_SIZE'],
                          bd_dilate_size=4,
                          mode=args['MODE'],
                          blend_images=args['BLEND_IMGS'],
                          robust_train=args['ROBUST_TRAIN'],
                          comp_mask=args['COMP_MASK'],
                          single_source=args['SINGLE_SOURCE'],
                          seed=args['SEED']
                          )
    val_dataset = WildFire(root=args['ROOTDATASET'],
                          list_path=args['VALIDSET'],
                          num_classes=args['NUM_CLASSES'],
                          multi_scale=False,
                          flip=False,
                          brightness=False,
                          ignore_label=args['IGNORE_LABEL'],
                          scale_factor=args['SCALE_FACTOR'],
                          crop_size=args['CROP_SIZE'],
                          base_size=args['BASE_SIZE'],
                          bd_dilate_size=4,
                          mode=args['MODE'],
                          blend_images=False,
                          robust_train=args['ROBUST_TRAIN'],
                          comp_mask=False,
                          single_source=False,
                          seed=args['SEED'])
    test_dataset = WildFire(root=args['ROOTDATASET'],
                        list_path=args['TESTSET'],
                        num_classes=args['NUM_CLASSES'],
                        multi_scale=False,
                        flip=False,
                        brightness=False,
                        ignore_label=args['IGNORE_LABEL'],
                        scale_factor=args['SCALE_FACTOR'],
                        crop_size=args['CROP_SIZE'],
                        base_size=args['BASE_SIZE'],
                        bd_dilate_size=4,
                        mode=args['MODE'],
                        blend_images=False,
                        robust_train=args['ROBUST_TRAIN'],
                        comp_mask=False,
                        single_source=False,
                        seed=args['SEED'])
    return train_dataset, val_dataset, test_dataset if test else ()

def get_model(args):
    channels = {'rgb':3, 'ir':1, 'fusion':4}
    if 'pidnet_s' == args['MODEL']:
        model = PIDNet(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=32, ppm_planes=96, head_planes=128, augment=True, channels=channels[args['MODE']])
    elif 'pidnet_m' == args['MODEL']:
        model = PIDNet(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=96, head_planes=128, augment=True, channels=channels[args['MODE']])
    elif 'pidnet_l' == args['MODEL']:
        model = PIDNet(m=3, n=4, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=112, head_planes=256, augment=True, channels=channels[args['MODE']])
    elif 'async2_s' == args['MODEL']:
        model = PIDnetTF(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=32, ppm_planes=96, head_planes=128, augment=True, channels=channels[args['MODE']], input_resolution=args['CROP_SIZE'], window_size=(args['WINDOW_SIZE'], args['WINDOW_SIZE']), tf_depths=args['TF_CONFIG'], robust_module=args['ROBUST_TRAIN'])
    elif 'async2_m' == args['MODEL']:
        model = PIDnetTF(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=96, head_planes=128, augment=True, channels=channels[args['MODE']], input_resolution=args['CROP_SIZE'], window_size=(args['WINDOW_SIZE'], args['WINDOW_SIZE']), tf_depths=args['TF_CONFIG'], robust_module=args['ROBUST_TRAIN'])
    
    if args['PRETRAINED'] is not None:
        model.imgnet_pretrain(args['PRETRAINED'])
    model.to(device=args['DEVICE'])
    return model