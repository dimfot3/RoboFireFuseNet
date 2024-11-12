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
        self.n_classes = args['NUM_CLASSES']
        self.bd_weight = args['BD_WEIGHT']
        self.class_weights = args['CLASS_WEIGHTS']
        self.defuse_weights = [1, 1, 1]
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
    
    def compute_Ldc(self, class_centers, rho_1=1):
        """
        Compute the Ldc loss function as described in the equation.
        
        Args:
            class_centers: A list of tensors containing the centers of modality-specific and modality-shared features.
                        The list should contain (rgb_shared_centers, ir_shared_centers, rgb_specific_centers, ir_specific_centers).
            rho_1: The regularization parameter.
            
        Returns:
            Ldc: The computed loss.
        """
        Csh_V = torch.stack(class_centers[0], dim=0)
        Csh_I = torch.stack(class_centers[1], dim=0)
        Csp_V = torch.stack(class_centers[2], dim=0)
        Csp_I = torch.stack(class_centers[3], dim=0)
        Ldc = 0
        for p in range(len(Csp_V)):  # Iterate over classes (identity)
            max_distance_V = torch.norm(Csp_V[p] - Csp_V, p=2, dim=1).max() - torch.norm(Csp_V[p] - Csh_V, p=2, dim=1).min() + rho_1
            max_distance_V = torch.max(max_distance_V, torch.tensor(0.0))  # Apply max with 0 to ensure non-negative
            max_distance_I = torch.norm(Csp_I[p] - Csp_I, p=2, dim=1).max() - torch.norm(Csp_I[p] - Csh_I, p=2, dim=1).min() + rho_1
            max_distance_I = torch.max(max_distance_I, torch.tensor(0.0))  # Apply max with 0 to ensure non-negative
            Ldc += max_distance_V + max_distance_I
        return Ldc

    def compute_Lsps(self, class_centers, rho_2):
        """
        Compute the Lsps loss function as described in the equation.
        
        Args:
            class_centers: A list of tensors containing the centers of modality-specific features.
                        The list should contain (rgb_specific_centers, ir_specific_centers).
            rho_2: The regularization parameter.
            
        Returns:
            Lsps: The computed loss.
        """
        rgb_specific_centers = torch.stack(class_centers[0], dim=0)
        ir_specific_centers = torch.stack(class_centers[1], dim=0)
        Lsps = 0
        # For each identity p
        for p in range(len(rgb_specific_centers)):  # Iterate over classes (identity)
            # Compute distances for modality V (RGB)
            distances_V = torch.norm(rgb_specific_centers[p] - rgb_specific_centers, p=2, dim=1)  # pairwise distances
            mask_V = torch.ones_like(distances_V, dtype=torch.bool)
            mask_V[p] = False  # Set the self-distance to be excluded
            
            # Apply the mask and find the minimum distance
            min_distance_V = distances_V[mask_V].min()  # Find the minimum distance to other identities
            Lsps += torch.max(rho_2 - min_distance_V, torch.tensor(0.0))  # Apply the max with 0
            # Compute distances for modality I (IR)
            distances_I = torch.norm(ir_specific_centers[p] - ir_specific_centers, p=2, dim=1)  # pairwise distances
            mask_I = torch.ones_like(distances_I, dtype=torch.bool)
            mask_I[p] = False  # Set the self-distance to be excluded
            min_distance_I = distances_I[mask_I].min()  # Find the minimum distance to other identities
            Lsps += torch.max(rho_2 - min_distance_I, torch.tensor(0.0))  # Apply the max with 0
        return Lsps
    
    def compute_Lshs(self, class_centers, alpha=2, rho_3=0.7):
        """
        Compute the Lshs loss function as described in the equation.
        
        Args:
            class_centers: A list of tensors containing the centers of modality-shared features.
                        The list should contain (rgb_shared_centers, ir_shared_centers).
            alpha: The weight for the self-distance term.
            rho_3: The regularization parameter.
            
        Returns:
            Lshs: The computed loss.
        """
        rgb_shared_centers = torch.stack(class_centers[0], dim=0)  # shape: (N_classes, C)
        ir_shared_centers = torch.stack(class_centers[1], dim=0)   # shape: (N_classes, C)

        Lshs = 0
        # For each identity p
        for p in range(len(rgb_shared_centers)):  # Iterate over classes (identity)
            
            # First term: self-distance (will always be 0 since it's Cpsh,V - Cpsh,V)
            self_distance_V = torch.norm(rgb_shared_centers[p] - rgb_shared_centers[p], p=2)  # Should be zero
            self_distance_I = torch.norm(ir_shared_centers[p] - ir_shared_centers[p], p=2)  # Should be zero
            Lshs += alpha * self_distance_V**2 + alpha * self_distance_I**2  # Weighted self-distance term

            # Compute distances for modality V (RGB)
            distances_V = torch.norm(rgb_shared_centers[p] - rgb_shared_centers, p=2, dim=1)  # pairwise distances
            mask_V = torch.ones_like(distances_V, dtype=torch.bool)
            mask_V[p] = False  # Set the self-distance to be excluded
            min_distance_V = distances_V[mask_V].min()  # Find the minimum distance to other identities
            Lshs += torch.max(rho_3 - min_distance_V, torch.tensor(0.0))  # Apply the max with 0
            
            # Compute distances for modality I (IR)
            distances_I = torch.norm(ir_shared_centers[p] - ir_shared_centers, p=2, dim=1)  # pairwise distances
            mask_I = torch.ones_like(distances_I, dtype=torch.bool)
            mask_I[p] = False  # Set the self-distance to be excluded
            min_distance_I = distances_I[mask_I].min()  # Find the minimum distance to other identities
            Lshs += torch.max(rho_3 - min_distance_I, torch.tensor(0.0))  # Apply the max with 0
        return Lshs

    def compute_class_centers(self, rgb_shared, ir_shared, rgb_specific, ir_specific, label_mask):
        """
        Compute the centers of each class for each of the four feature maps: 
        RGB shared, IR shared, RGB specific, IR specific.
        
        Args:
        rgb_shared: Tensor of shape (batch_size, height, width, channels) for RGB shared features
        ir_shared: Tensor of shape (batch_size, height, width, channels) for IR shared features
        rgb_specific: Tensor of shape (batch_size, height, width, channels) for RGB-specific features
        ir_specific: Tensor of shape (batch_size, height, width, channels) for IR-specific features
        label_mask: Tensor of shape (batch_size, height, width) with class labels for each pixel
        N_classes: Integer, the number of classes in the dataset
        
        Returns:
        class_centers: List of tensors, where each tensor has shape (N_classes, feature_dimension) 
                    representing the class centers for each feature map.
        """
        label_mask = F.interpolate(label_mask.unsqueeze(1).to(torch.float), rgb_shared.shape[-2:], mode='nearest')
        feature_maps = [rgb_shared, ir_shared, rgb_specific, ir_specific]
        class_centers = []
        for fi, feature_map in enumerate(feature_maps):
            feature_centers = []
            for class_id in torch.unique(label_mask):
                if class_id == self.ignore_label: continue
                class_mask = (label_mask == class_id).float()
                class_features = feature_map * class_mask
                class_center = class_features.sum(dim=(0, 2, 3))
                class_pixel_count = class_mask.sum(dim=(0, 2, 3))
                class_center /= (class_pixel_count + 1e-6)
                feature_centers.append(class_center)
            class_centers.append(feature_centers)
        return class_centers

    def get_loss(self, outputs, labels, bd_gt):
        """
        Calculates total prediction loss, including semantic loss (using OHEM or cross-entropy), 
        boundary loss (using CE), and additional semantic loss for detected boundaries.
        :return: the loss, the semantic maps (from both propotion and integral head),
        the avg pixel accuracy and the segmentation and boundary losses.
        """
        defuse_loss = 0
        if(len(outputs) > 3):
            intermed_feat, outputs = outputs[-1],  outputs[:-1]
            class_centers = self.compute_class_centers(*intermed_feat, labels)
            l1 = self.compute_Ldc(class_centers, rho_1=1)
            l2 = self.compute_Lsps(class_centers, rho_2=0.7)
            l3 = self.compute_Lshs(class_centers, alpha=2, rho_3=0.7)
            defuse_loss = sum([self.defuse_weights[0] * l1 + self.defuse_weights[1] * l2 + self.defuse_weights[2] * l3])
            
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
        loss = loss_s + loss_b + (loss_sb if not torch.isnan(loss_sb) else 0) + defuse_loss
        return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s, loss_b]


class MaskedMSELoss():
    def __init__(self, args):
        return

    def get_loss(self, pred, target, mask):
        loss = ((pred - target) ** 2) * mask
        loss = loss.reshape(loss.size(0), -1).mean(dim=1)
        return loss

