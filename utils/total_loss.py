import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

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

class LovaszCE:
    def __init__(self, cls_weights):
        self.cls_weights = cls_weights

    def lovasz_softmax_loss(self, outputs, labels, classes='all', per_image=False, ignore=None):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """
        probas = F.softmax(outputs, dim=1)
        if per_image:
            loss = self.mean(self.lovasz_softmax_flat(*(self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore)), classes=classes)
                            for prob, lab in zip(probas, labels))
        else:
            loss = self.lovasz_softmax_flat(*(self.flatten_probas(probas, labels, ignore)), classes=classes)
        return loss


    def lovasz_softmax_flat(self, probas, labels, classes='all'):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float() # foreground for class c
            if (classes == 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, self.lovasz_grad(fg_sorted)) * self.cls_weights[c])
        return self.mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)

        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels

    def isnan(self, x):
        return x != x

    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(self.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n
    
    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1: # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

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
        self.n_classes = args['NUM_CLASSES']
        self.bd_weight = args['BD_WEIGHT']
        self.class_weights = args['CLASS_WEIGHTS']
        self.defuse_weights = [1, 0.5, 0.5]
        self.miou_ce = LovaszCE(self.class_weights)
        self.mse_loss = nn.MSELoss()
        self.loss_params = torch.nn.SmoothL1Loss()
    
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
        loss = torch.tensor(0).to(labels.device).to(torch.float)
        h, w = labels.size(1), labels.size(2)
        ph, pw = outputs[0].size(2), outputs[0].size(3)
        if (ph != h) or (pw != w):
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(outputs[i], size=(
                    h, w), mode='bilinear', align_corners=self.align_corners)
        acc  = self.pixel_acc(outputs[-2], labels)
        loss_s = self.miou_ce.lovasz_softmax_loss(outputs[1], labels, classes='all', per_image=False) + \
            self.miou_ce.lovasz_softmax_loss(outputs[0], labels, classes='all', per_image=False) 
        loss_b = self.bd_criterion(outputs[-1], bd_gt)      # the boundary loss l1
        filler = torch.ones_like(labels) * self.ignore_label
        bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>self.t_thresh_bd, labels, filler)
        loss_sb = self.sem_criterion(outputs[-2], bd_label)
        loss += (loss_s if not torch.isnan(loss_sb) else 0)  + (loss_b if not torch.isnan(loss_b) else 0) + (loss_sb if not torch.isnan(loss_sb) else 0)
        return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s, loss_b]

