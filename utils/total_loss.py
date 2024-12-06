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

def kv_loss(student_features, teacher_features, temperature=1.0):
    """
    Compute Knowledge Distillation (KV Loss) between student and teacher features.
    
    Args:
        student_features (torch.Tensor): Features from the RGB-transformed model (student).
        teacher_features (torch.Tensor): Features from the IR model (teacher).
        temperature (float): Temperature parameter to smooth the distributions.

    Returns:
        torch.Tensor: The KV loss value.
    """
    # Normalize features for cosine similarity
    student_features = student_features / (student_features.norm(dim=1, keepdim=True) + 1e-8)
    teacher_features = teacher_features / (teacher_features.norm(dim=1, keepdim=True) + 1e-8)

    # Compute Cosine Similarity loss
    cosine_similarity_loss = 1 - torch.mean((student_features * teacher_features).sum(dim=1))

    # Optionally add MSE as another component (optional)
    mse_loss = torch.nn.functional.mse_loss(student_features, teacher_features)

    # Combine the two losses (optional: weighted sum)
    loss = cosine_similarity_loss + (temperature * mse_loss)

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
        self.miou_ce = NewCE(self.class_weights)
        self.mse_loss = nn.MSELoss()
        self.loss_params = torch.nn.SmoothL1Loss()
        

        self.affine_loss = lambda tf_pred, tf_gt: torch.nn.functional.mse_loss(tf_pred, tf_gt)
        self.transformation_consistency_loss = lambda x_ir_new, x_ir, tf: torch.nn.functional.mse_loss(
            F.grid_sample(x_ir_new, F.affine_grid(tf, x_ir_new.size(), align_corners=False), align_corners=False),
            x_ir
        )
        self.cosine_similarity_loss = lambda x_ir_new, x_rgb_ir: 1 - torch.nn.functional.cosine_similarity(
            x_ir_new, x_rgb_ir, dim=-1
        ).mean()
        self.mse_loss = lambda x_ir_new, x_rgb_ir: torch.nn.functional.mse_loss(x_ir_new, x_rgb_ir)
        self.kl_divergence_loss = lambda x_ir_new, x_ir: kv_loss(x_ir_new, x_ir, temperature=1)
        self.geometric_alignment_loss = lambda x_ir_new, x_ir, tf_pred: torch.nn.functional.mse_loss(
            F.grid_sample(x_ir, F.affine_grid(tf_pred, x_ir.size(), align_corners=False), align_corners=False),
            x_ir_new
        )

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
        This pushes the maximum distances between modality shared features furhter than minimum distances between 
        modality shared and spcecific features for same id.
        
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
            max_distance_V = torch.max(max_distance_V, torch.tensor(0.0, requires_grad=True))
            max_distance_I = torch.norm(Csp_I[p] - Csp_I, p=2, dim=1).max() - torch.norm(Csp_I[p] - Csh_I, p=2, dim=1).min() + rho_1
            max_distance_I = torch.max(max_distance_I, torch.tensor(0.0, requires_grad=True))
            Ldc += max_distance_V + max_distance_I
        return Ldc

    def compute_Lsps(self, class_centers, rho_2):
        """
        Pushes away different-id features in modality specific features.
        
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
            Lsps += torch.max(rho_2 - min_distance_V, torch.tensor(0.0, requires_grad=True))  # Apply the max with 0
            # Compute distances for modality I (IR)
            distances_I = torch.norm(ir_specific_centers[p] - ir_specific_centers, p=2, dim=1)  # pairwise distances
            mask_I = torch.ones_like(distances_I, dtype=torch.bool)
            mask_I[p] = False  # Set the self-distance to be excluded
            min_distance_I = distances_I[mask_I].min()  # Find the minimum distance to other identities
            Lsps += torch.max(rho_2 - min_distance_I, torch.tensor(0.0, requires_grad=True))  # Apply the max with 0
        return Lsps
    
    def compute_Lshs(self, class_centers, alpha=2, rho_3=0.7):
        """
        Bring closer the features for same id between the modality-share features. Push away modality share features with different id.
        
        Args:
            class_centers: A list of tensors containing the centers of modality-shared features.
                        The list should contain (rgb_shared_centers, ir_shared_centers).
            alpha: The weight for the self-distance term.
            rho_3: The regularization parameter.
            
        Returns:
            Lshs: The computed loss.
        """
        rgb_shared_centers = torch.stack(class_centers[0], dim=0)
        ir_shared_centers = torch.stack(class_centers[1], dim=0)
        total_shared_centers = torch.cat([rgb_shared_centers, ir_shared_centers], dim=0)
        Lshs = 0
        # For each identity p
        for p in range(len(rgb_shared_centers)):  # Iterate over classes (identity)
            
            # First term: self-distance (will always be 0 since it's Cpsh,V - Cpsh,V)
            self_distance = torch.norm(rgb_shared_centers[p] - ir_shared_centers[p], p=2)  # Should be zero
            Lshs += alpha * self_distance**2 

            # Compute distances for modality V (RGB)
            distances_V = torch.norm(rgb_shared_centers[p] - total_shared_centers, p=2, dim=1)  # pairwise distances
            mask_V = torch.ones_like(distances_V, dtype=torch.bool)
            mask_V[p] = False  # Set the self-distance to be excluded
            mask_V[2*p] = False  # Set the self-distance to be excluded
            min_distance_V = distances_V[mask_V].min()  # Find the minimum distance to other identities
            Lshs += torch.max(rho_3 - min_distance_V, torch.tensor(0.0, requires_grad=True))  # Apply the max with 0
            
            # Compute distances for modality I (IR)
            distances_I = torch.norm(ir_shared_centers[p] - total_shared_centers, p=2, dim=1)  # pairwise distances
            mask_I = torch.ones_like(distances_I, dtype=torch.bool)
            mask_I[p] = False  # Set the self-distance to be excluded
            mask_I[2*p] = False  # Set the self-distance to be excluded
            min_distance_I = distances_I[mask_I].min()  # Find the minimum distance to other identities
            Lshs += torch.max(rho_3 - min_distance_I, torch.tensor(0.0, requires_grad=True))  # Apply the max with 0
        return Lshs

    def compute_class_centers(self, rgb_shared, ir_shared, rgb_specific, ir_specific, label_mask):
        """
        Compute the centers of each class for each of the four feature maps: 
        RGB shared, IR shared, RGB specific, IR specific, based on pairwise distances.
        
        Args:
        rgb_shared: Tensor of shape (batch_size, channels, height, width) for RGB shared features
        ir_shared: Tensor of shape (batch_size, channels, height, width) for IR shared features
        rgb_specific: Tensor of shape (batch_size, channels, height, width) for RGB-specific features
        ir_specific: Tensor of shape (batch_size, channels, height, width) for IR-specific features
        label_mask: Tensor of shape (batch_size, height, width) with class labels for each pixel
        
        Returns:
        class_centers: List of tensors, where each tensor has shape (N_classes, channels) 
                    representing the class centers for each feature map.
        """
        # Ensure the label mask matches the spatial dimensions of the features
        label_mask = F.interpolate(label_mask.unsqueeze(1).float(), rgb_shared.shape[2:], mode='nearest').squeeze(1)
        feature_maps = [rgb_shared, ir_shared, rgb_specific, ir_specific]
        class_centers = []
        for feature_map in feature_maps:
            feature_centers = []
            batch_size, channels, height, width = feature_map.shape
            feature_map = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)  # (B*H*W, C)
            label_mask_flat = label_mask.view(-1)  # (B*H*W)            
            for class_id in torch.unique(label_mask_flat):
                if class_id == self.ignore_label:   continue
                class_pixels = feature_map[label_mask_flat == class_id]  # (N_pixels, C)
                # distances = torch.cdist(class_pixels, class_pixels, p=2)
                # total_distances = distances.sum(dim=1)
                central_pixel = class_pixels.median(dim=0).values #class_pixels[total_distances.argmin()]
                feature_centers.append(central_pixel)
            class_centers.append(feature_centers)
        return class_centers

    def get_loss(self, outputs, labels, bd_gt, tf=None):
        """
        Calculates total prediction loss, including semantic loss (using OHEM or cross-entropy), 
        boundary loss (using CE), and additional semantic loss for detected boundaries.
        :return: the loss, the semantic maps (from both propotion and integral head),
        the avg pixel accuracy and the segmentation and boundary losses.
        """
        loss = torch.tensor(0).to(labels.device).to(torch.float)
        robust_out, outputs = outputs[-1], outputs[:-1]
        if tf != None:
            x_ir, x_ir_new, x_rgb_ir, tf_pred = robust_out
            maskout = torch.repeat_interleave(F.interpolate(torch.tensor((labels == self.ignore_label).to(torch.float)).unsqueeze(1), \
                                    x_ir_new.shape[-2:], mode='nearest').to(torch.bool), x_rgb_ir.shape[1], 1)
            x_rgb_ir, x_ir_new, x_ir = x_rgb_ir * (~maskout), x_ir_new * (~maskout), x_ir * (~maskout)
            loss +=   self.kl_divergence_loss(x_rgb_ir, x_ir)
            #         0.0 * self.transformation_consistency_loss(x_ir_new, x_ir, tf) + \
            #         0.0 * self.cosine_similarity_loss(x_ir_new, x_rgb_ir) + \
            #         0.0 * self.mse_loss(x_ir_new, x_rgb_ir) + \
            #         0.0 * self.kl_divergence_loss(x_ir, x_ir_new) +\
            #         0.0 * self.geometric_alignment_loss(x_ir_new, x_ir, tf_pred) +\
            #         0.0 * self.affine_loss(tf_pred[:, :, 2], tf[:, :, 2])
        h, w = labels.size(1), labels.size(2)
        ph, pw = outputs[0].size(2), outputs[0].size(3)
        if (ph != h) or (pw != w):
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(outputs[i], size=(
                    h, w), mode='bilinear', align_corners=self.align_corners)
        acc  = self.pixel_acc(outputs[-2], labels)
        # loss_s = self.sem_criterion(outputs[:-1], labels)
        loss_s = self.miou_ce.lovasz_softmax_loss(outputs[1], labels, classes='all', per_image=False) + \
            self.miou_ce.lovasz_softmax_loss(outputs[0], labels, classes='all', per_image=False) 
        # loss_miou = self.miou_ce.lovasz_softmax_loss(outputs[1], labels, classes='all', per_image=False)
        loss_b = self.bd_criterion(outputs[-1], bd_gt)      # the boundary loss l1
        filler = torch.ones_like(labels) * self.ignore_label
        bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>self.t_thresh_bd, labels, filler)
        loss_sb = self.sem_criterion(outputs[-2], bd_label)
        loss += (loss_s if not torch.isnan(loss_sb) else 0)  + (loss_b if not torch.isnan(loss_sb) else 0) + (loss_sb if not torch.isnan(loss_sb) else 0) # + loss_miou
        return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s, loss_b]


class MaskedMSELoss():
    def __init__(self, args):
        self.bd_criterion = BondaryLoss(coeff_bce=1)
        self.ignore_label = 255

    def get_loss(self, outputs, target, mask, edges):
        h, w = target.size(-2), target.size(-1)
        ph, pw = outputs[0].size(2), outputs[0].size(3)
        if (ph != h) or (pw != w):
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(outputs[i], size=(
                    h, w), mode='bilinear', align_corners=self.align_corners)
                
        pred = outputs[1]
        integ_pred = outputs[0]
        bd_pred = outputs[2]
        loss_sem = (((pred - target) ** 2) * mask).reshape(-1).mean()
        loss_int = (((integ_pred - target) ** 2) * mask).reshape(-1).mean()
        loss_b = (self.bd_criterion(outputs[-1], edges)).reshape(-1).mean()
        bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>0.8, edges, 0).unsqueeze(1)
        loss_sb = (((bd_label - bd_pred) ** 2) * ((outputs[-1][:,0,:,:])>0.8) * mask[:, 0, :, :]).reshape(-1).mean()
        loss = loss_sem + loss_int + loss_b + (loss_sb if not torch.isnan(loss_sb) else 0)
        return loss.unsqueeze(0)


class NewCE:
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