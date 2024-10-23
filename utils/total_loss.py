import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class CrossEntropy(nn.Module):
    def __init__(self, args, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.Tensor(weight).to(args['DEVICE']),
            ignore_index=ignore_label
        )
        self.num_outputs = args['NUM_OUTPUTS']
        self.balance_weights = args['BALANCE_WEIGHTS']
        self.sb_weights = args['SB_WEIGHTS']

    def _forward(self, score, target):
        loss = self.criterion(score, target)
        return loss

    def forward(self, score, target):
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]
        balance_weights = self.balance_weights
        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        elif len(score) == 1:
            return self.sb_weights * self._forward(score[0], target)
        else:
            raise ValueError("lengths of prediction and target are not identical!")

class OhemCrossEntropy(nn.Module):
    """
    This is an implementation of Online Hard Example Mining and 
    is alternative of CE that can propably increase convergence 
    speed.
    """
    def __init__(self, args, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.Tensor(weight).to(args['DEVICE']),
            ignore_index=ignore_label,
            reduction='none'
        )
        self.num_outputs = args['NUM_OUTPUTS']
        self.balance_weights = args['BALANCE_WEIGHTS']
        self.sb_weights = args['SB_WEIGHTS']

    def _ce_forward(self, score, target):
        loss = self.criterion(score, target)
        return loss

    def _ohem_forward(self, score, target, **kwargs):
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        if pred.numel() == 0:   return torch.tensor(0.0)
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]
        balance_weights = self.balance_weights
        sb_weights = self.sb_weights
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        else:
            raise ValueError("lengths of prediction and target are not identical!")

class BondaryLoss(nn.Module):
    """
    Binary cross-entropy, evaluating weights for positives and negatives per batch.
    """
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def weighted_bce(self, bd_pre, target):
        n, c, h, w = bd_pre.size()
        log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
        target_t = target.view(1, -1)
        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        weight = torch.zeros_like(log_p).float()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num.float() + neg_num
        weight[pos_index] = neg_num.float() * 1.0 / sum_num
        weight[neg_index] = pos_num.float() * 1.0 / sum_num
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight.float(), reduction='mean')
        return loss

    def forward(self, bd_pre, bd_gt):
        bce_loss = self.coeff_bce * self.weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        return loss
    
class TotalLoss:
    def __init__(self, args):
        self.align_corners = args['ALIGN_CORNERS']
        self.ignore_label = args['IGNORE_LABEL']
        self.t_thresh_bd = args['T_THRESH_BDLOSS']
        self.bd_weight = args['BD_WEIGHT']
        self.class_weights = args['CLASS_WEIGHTS']
        if args['USE_OHEM']:
            self.sem_criterion = OhemCrossEntropy(args, ignore_label=args['IGNORE_LABEL'],
                                        thres=args['OHEMTHRES'],
                                        min_kept=args['OHEMKEEP'],
                                        weight=args['CLASS_WEIGHTS'])
        else:
            self.sem_criterion = CrossEntropy(args, ignore_label=args['IGNORE_LABEL'],
                                    weight=args['CLASS_WEIGHTS'])
        self.bd_criterion = BondaryLoss(coeff_bce=self.bd_weight)
        
    
    def pixel_acc(self, pred, label):
        """
        Calculates the mean pixel accuracy for valid pixels (non-negative labels) across the batch.
        """
        _, preds = torch.max(pred, dim=1)
        valid = (label != self.ignore_label).long()
        pixel_sum = torch.sum(valid)
        acc_sum = torch.sum(valid * (preds == label).long())
        acc = (acc_sum.float() / (pixel_sum.float() + 1e-10)).detach().cpu().numpy()
        acc_per_class = [0] * len(self.class_weights)
        for i, cls in enumerate(self.class_weights):
            cls_vld = (label == i).long()
            pixel_sum = torch.sum(cls_vld)
            acc_sum = torch.sum(cls_vld * (preds == label).long())
            acc_per_class[i] = (acc_sum.float() / (pixel_sum.float() + 1e-10)).detach().cpu().numpy()
        acc = np.array([acc, *acc_per_class])
        return acc

    def get_loss(self, outputs, labels, bd_gt):
        """
        Calculates total prediction loss, including semantic loss (using OHEM or cross-entropy), 
        boundary loss (using CE), and additional semantic loss for detected boundaries.
        :return: the loss, the semantic maps (from both propotion and integral head),
        the avg pixel accuracy and the segmentation and boundary losses.
        """
        h, w = labels.size(1), labels.size(2)
        ph, pw = outputs[0].size(2), outputs[0].size(3)
        if (ph != h) or (pw != w):
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(outputs[i], size=(
                    h, w), mode='bilinear', align_corners=self.align_corners)

        acc  = self.pixel_acc(outputs[-2], labels)
        loss_s = self.sem_criterion(outputs[:-1], labels)   # the semantic proportion and integral l0, l2
        loss_b = self.bd_criterion(outputs[-1], bd_gt)      # the boundary loss l1
        filler = torch.ones_like(labels) * self.ignore_label
        bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>self.t_thresh_bd, labels, filler)
        loss_sb = self.sem_criterion(outputs[-2], bd_label)
        loss = loss_s + loss_b + (loss_sb if not torch.isnan(loss_sb) else 0)
        return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s, loss_b]


class MaskedMSELoss():
    def __init__(self, args):
        return

    def get_loss(self, pred, target, mask):
        loss = ((pred - target) ** 2) * mask
        loss = loss.reshape(loss.size(0), -1).mean(dim=1)
        return loss

if __name__ == '__main__':
    args = {'USE_OHEM': True,
            'DEVICE': 'cpu',
            'OHEMTHRES': 0.9,
            'OHEMKEEP': 131072,
            'CLASS_WEIGHTS': [0.3, 0.3, 0.4],
            'BALANCE_WEIGHTS': [0.4, 1.0],
            'SB_WEIGHTS': 1.0,
            'ALIGN_CORNERS': True,
            'IGNORE_LABEL': 255,
            'NUM_OUTPUTS': 2,
            'T_THRESH_BDLOSS': 0.8,
            'BD_WEIGHT': 1
            }
    totalloss = TotalLoss(args)
    propotion_head = torch.tensor([[
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
    [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
], [
    [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
    [[1.0, 0.9, 0.8], [0.7, 0.6, 0.5], [0.4, 0.3, 0.2]],
    [[0.6, 0.6, 0.6], [0.6, 0.6, 0.6], [0.6, 0.6, 0.6]]
]]).to(args['DEVICE'])

    integral_head = torch.tensor([[
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
        [[1.0, 0.9, 0.8], [0.7, 0.6, 0.5], [0.4, 0.3, 0.2]],
        [[0.6, 0.6, 0.6], [0.6, 0.6, 0.6], [0.6, 0.6, 0.6]]
    ], [
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        [[1.0, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    ]]).to(args['DEVICE'])

    # Create dummy labels with specific values
    labels = torch.tensor([
        [[0, 1, 2], [2, 1, 0], [0, 1, 2]],
        [[2, 255, 0], [0, 1, 2], [2, 1, 0]]
    ]).long().to(args['DEVICE'])

    # Create dummy boundary ground truth with specific values
    bd_gt = torch.tensor([
        [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    ]).float().to(args['DEVICE'])

    bd_head = torch.tensor([
        [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    ]).float().to(args['DEVICE']).unsqueeze(1)
    
    # Combine the heads into the outputs dictionary
    outputs = [propotion_head, integral_head, bd_head]
    loss, outseg, acc, [loss_s, loss_b] = totalloss.get_loss(outputs, labels, bd_gt)
    np.testing.assert_almost_equal(acc, (outputs[-2].argmax(1)==labels).sum() / 17)