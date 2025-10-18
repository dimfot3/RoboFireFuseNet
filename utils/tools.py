import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os 
import argparse
import yaml
import random
from skimage.measure import label as label2
from skimage.measure import regionprops
from scipy.ndimage import label as label1


def set_reproducibility(seed):
    """
    Set the random seed for reproducibility in experiments.

    Parameters:
    seed (int): The seed value to set for reproducibility.

    Returns:
    None
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8' # or ':16:8' 16 or 4096 is mb of space for cublas
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def str2bool(v):
    """
    Convert a string representation of a boolean to a boolean value.

    Parameters:
    v (str or bool): The input value to convert.

    Returns:
    bool: The corresponding boolean value.

    Raises:
    argparse.ArgumentTypeError: If the input is not a valid boolean string.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """
    Parse command-line arguments for training parameters.

    Returns:
    dict: A dictionary containing the configuration parameters from 
          the YAML file and command-line arguments.

    Description:
    - This function uses argparse to handle command-line input and 
      reads a YAML configuration file. 
    - Command-line arguments include learning rate, batch size, 
      weight decay, session name, number of epochs, device, stop 
      counter, and online logging options.
    """

    parser = argparse.ArgumentParser(description='Setting the training parameters')
    parser.add_argument('--yaml_file', type=str, help='Path to YAML file', default='wildfire.yaml')
    parser.add_argument('--LR', type=float, help='Learning Rate')
    parser.add_argument('--CHECKPOINT', type=str, help='Checkpoint')
    parser.add_argument('--BATCHSIZE', type=int, help='Batch Size')
    parser.add_argument('--WD', type=float, help='Weight decay')
    parser.add_argument('--SESSIONAME', type=str, help='Session Name')
    parser.add_argument('--EPOCHS', type=int, help='Number of Epochs')
    parser.add_argument('--DEVICE', type=str, help='Device "cpu" or "cuda"')
    parser.add_argument('--STOPCOUNTER', type=int, help='Stop counter')
    parser.add_argument('--ONLINELOG', type=str2bool, help='Online Log in weight and biases')
    parser.add_argument('--PRETRAINED', type=str, help='full path to pretrain weights')
    parser.add_argument('--OPTIM', type=str, help='Optimizer')
    parser.add_argument('--SCHED', type=str, help='Scheduler')
    
    args = parser.parse_args()
    args = {key: value for key, value in vars(args).items() if value is not None}
    with open('config/' + args['yaml_file'], 'r') as file:
        config = yaml.safe_load(file)
    for key, value in args.items():
        if key in config:
            config[key] = value
    return config


def plot_instance_masks(gt_instance_mask, pred_instance_mask, gt_class, pred_major_class, instance_id):
    """Helper function to plot the ground truth and predicted instance masks."""
    
    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gt_instance_mask.reshape(254, 254), cmap='gray')
    axes[0].set_title(f'Ground Truth Instance (Class {gt_class}) - ID {instance_id}')
    axes[0].axis('off')

    
    plt.show()

def process_mask(mask, k):
    mask = mask.detach().cpu().numpy()
    # Initialize the output mask
    output_mask = np.zeros_like(mask)
    
    # Iterate over classes 1 and 2
    for class_id in [1, 2]:
        # Create a binary mask for the current class
        class_mask = (mask == class_id).astype(np.uint8)
        
        # Label the connected components (instances)
        labeled_mask = label2(class_mask).reshape(mask.shape)
        # Get properties of labeled regions
        regions = regionprops(labeled_mask)
        
        for region in regions:
            if region.area >= k:
                # Keep the instance in the output mask
                output_mask[labeled_mask == region.label] = class_id
                
    return output_mask

def get_confusion_matrix_instancewise(seg_gt, output, num_class, k=0.5, ignore=255):
    """
    Compute the confusion matrix for a segmentation task based on instance-level accuracy.

    Parameters:
    seg_gt (torch.Tensor): Ground truth segmentation map with shape (N, H, W).
    output (torch.Tensor): Model output predictions with shape (N, num_class, H, W).
    num_class (int): The number of classes in the segmentation task.
    k (float): Threshold for considering an instance as True Positive (percentage overlap).
    ignore (int, optional): Class index to ignore in the computation. Defaults to 255.

    Returns:
    numpy.ndarray: Confusion matrix of shape (num_class, num_class).
    """
    # Convert tensors to long and find the prediction
    seg_gt = seg_gt.to(torch.long)
    seg_pred = output.argmax(dim=1).to(torch.long)
    # Create confusion matrix
    confusion_matrix = np.zeros((num_class, num_class), dtype=np.int64)
    img_shape = (seg_gt.shape[-2], seg_gt.shape[-1])
    for i in range(seg_gt.shape[0]):  # Iterate over batch
        gt_mask = seg_gt[i]
        pred_mask = seg_pred[i]

        # Ignore specified class (if necessary)
        valid_mask = gt_mask != ignore
        gt_mask = gt_mask[valid_mask].reshape(img_shape).cpu()
        pred_mask = pred_mask[valid_mask].reshape(img_shape).cpu()
        labeled_gt, num_gt_instances = label1(gt_mask)  # Get instances in GT
        for gt_instance_id in range(0, num_gt_instances + 1):  # Skip background (ID 0)
            # Get binary mask for this ground truth instance
            gt_instance_mask = (labeled_gt == gt_instance_id).reshape(img_shape)
            gt_class = np.unique(gt_mask[gt_instance_mask])  # Should be one class per instance
            gt_class = gt_class[0]  # Since it's one instance, there should be a single class

            # Get the predicted labels in the same area
            pred_instance_mask = pred_mask[gt_instance_mask]

            # Find the most common class in the predicted instance mask
            pred_class, counts = np.unique(pred_instance_mask, return_counts=True)
            pred_major_class = pred_class[np.argmax(counts)]

            # Calculate overlap ratio for the predicted class vs ground truth
            intersection = (pred_instance_mask == gt_class).sum()
            union = gt_instance_mask.sum()

            overlap_ratio = intersection / float(union)
            # plot_instance_masks(gt_instance_mask, pred_instance_mask, gt_class, pred_major_class, gt_instance_id)

            # Decide if it's a True Positive, False Positive, or False Negative
            if overlap_ratio >= k:
                confusion_matrix[gt_class, gt_class] += 1  # True Positive
            else:
                confusion_matrix[gt_class, pred_major_class] += 1  # False Positive or FN
                
    return confusion_matrix

def get_confusion_matrix(seg_gt, output, num_class, ignore=255):
    """
    Compute the confusion matrix for a segmentation task.

    Parameters:
    seg_gt (torch.Tensor): Ground truth segmentation map with shape (N, H, W).
    output (torch.Tensor): Model output predictions with shape (N, num_class, H, W).
    num_class (int): The number of classes in the segmentation task.
    ignore (int, optional): Class index to ignore in the computation. Defaults to 255.

    Returns:
    numpy.ndarray: Confusion matrix of shape (num_class, num_class).
    """
    seg_gt = seg_gt.to(torch.long)
    seg_pred = output.argmax(dim=1).to(torch.long)
    valid_mask = seg_gt != ignore
    seg_gt = seg_gt[valid_mask]
    seg_pred = seg_pred[valid_mask]
    index = seg_gt * num_class + seg_pred
    confusion_matrix = torch.bincount(index, minlength=num_class**2, weights=None).reshape(num_class, num_class).detach().cpu().numpy()
    return confusion_matrix

def calculate_metrics(confusion_matrix, runloss, cls_names = None, cls_weights=None, val=False):
    """
    Calculate various evaluation metrics from the confusion matrix.

    Parameters:
    confusion_matrix (numpy.ndarray): The confusion matrix of shape (num_classes, num_classes).
    runloss (float): The loss value for the current run.
    cls_names (list, optional): List of class names corresponding to the classes. Defaults to None.
    cls_weights (numpy.ndarray, optional): Weights for each class used in weighted metrics. Defaults to None.
    val (bool, optional): Indicator for validation metrics. Defaults to False.

    Returns:
    dict: A dictionary containing precision, recall, F1 score, accuracy, IoU, and other metrics.
    """
    num_classes = confusion_matrix.shape[0]
    # Initialize arrays to hold per class metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    accuracy = np.zeros(num_classes)
    iou = np.zeros(num_classes)
    # Total number of samples
    total_samples = np.sum(confusion_matrix)
    total_TN = 0
    # Loop over each class to compute precision, recall, f1-score, and accuracy
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = total_samples - (TP + FP + FN)
        total_TN += TN
        
        precision[i] = TP / (TP + FP) if (TP + FP) != 0 else 1
        recall[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0
        accuracy[i] = (TP + TN) / total_samples if total_samples != 0 else 1
        iou[i] = TP / (TP + FN + FP) if (TP + FN + FP) != 0 else 0

    # Calculate average metrics
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1_score)
    avg_acc = np.mean(accuracy)
    total_acc = np.sum(np.diag(confusion_matrix)) / total_samples
    miou = np.mean(iou)
    metrics = {
        f"{'val ' if val else ''}avg_f1": avg_f1,
        f"{'val ' if val else ''}avg_acc": avg_acc,
        f"{'val ' if val else ''}total_acc": total_acc,
        f"{'val ' if val else ''}avg_precision": avg_precision,
        f"{'val ' if val else ''}avg_recall": avg_recall,
        f"{'val ' if val else ''}miou": miou,
        f"{'val ' if val else ''}avgloss": runloss}
    names = np.arange(num_classes) if (cls_names == None) else cls_names
    cls_weights = np.ones(num_classes) if (cls_weights == None) else cls_weights / np.sum(cls_weights)
    metrics[f'{"val " if val else ""}weighted_f1'] = 0
    for i in range(num_classes):
        metrics[f'{"val " if val else ""}precision {names[i]}'] = precision[i]
        metrics[f'{"val " if val else ""}recall {names[i]}'] = recall[i]
        metrics[f'{"val " if val else ""}f1_score {names[i]}'] = f1_score[i]
        metrics[f'{"val " if val else ""}iou {names[i]}'] = iou[i]
        metrics[f'{"val " if val else ""}accuracy {names[i]}'] = accuracy[i]
        metrics[f'{"val " if val else ""}weighted_f1'] += cls_weights[i] * f1_score[i]
    return metrics, iou

def qualitive_eval(inf_model, val_data, ex_path='./outputs', name='example.png'):
    """
    Perform qualitative evaluation of the segmentation model and save the results.

    Parameters:
    inf_model (torch.nn.Module): The model used for inference.
    val_data (Dataset): The validation dataset containing images and labels.
    ex_path (str, optional): The directory path where output images will be saved. Defaults to './outputs'.
    name (str, optional): The filename for the saved output image. Defaults to 'example.png'.

    Returns:
    None
    """
    valid_loader = iter(DataLoader(val_data, batch_size=1, shuffle=True))
    f, ax = plt.subplots(2, 5, figsize=(20, 5))
    for sampleid in range(10):
        batch = next(valid_loader)
        images = batch[0]
        label = batch[1][0].detach().cpu().numpy().astype('uint8')
        outputs = inf_model(images)
        images = (np.transpose(images[0, :3, :, :].detach().cpu().numpy(), (1, 2, 0)) * val_data.std_rgb) + val_data.mean_rgb
        images = (images * 255).astype('uint8')
        outputs = outputs.detach().cpu().numpy().astype('uint8')[0]
        non_bg_idxs = outputs!=0
        outputs = val_data.label2color(outputs)
        images[non_bg_idxs] = 0.199 * images[non_bg_idxs] + 0.799 * outputs[non_bg_idxs]
        ax[sampleid // 5][sampleid % 5].imshow(images, aspect='auto')
    os.makedirs(ex_path, exist_ok=True)
    plt.savefig(os.path.join(ex_path, name))
    plt.close(f)

def qualitive_test(inf_model, val_data, ex_path='./outputs'):
    """
    Perform qualitative evaluation of the segmentation model and save the results.

    Parameters:
    inf_model (torch.nn.Module): The model used for inference.
    val_data (Dataset): The validation dataset containing images and labels.
    ex_path (str, optional): The directory path where output images will be saved. Defaults to './outputs'.
    name (str, optional): The filename for the saved output image. Defaults to 'example.png'.

    Returns:
    None
    """
    valid_loader = iter(DataLoader(val_data, batch_size=1, shuffle=True))
    os.makedirs(ex_path, exist_ok=True)
    for sampleid in range(len(valid_loader)):
        batch = next(valid_loader)
        images, name = batch[0], batch[3]
        label = batch[1][0].detach().cpu().numpy().astype('uint8')
        outputs = inf_model(images)
        images = (np.transpose(images[0, :3, :, :].detach().cpu().numpy(), (1, 2, 0)) * val_data.std_rgb) + val_data.mean_rgb
        images = (images * 255).astype('uint8')
        outputs = outputs.detach().cpu().numpy().astype('uint8')[0]
        non_bg_idxs = outputs!=0
        fire_instances = outputs == 2
        outputs = val_data.label2color(outputs)
        if val_data.num_classes == 3:
            outputs[fire_instances, 1:] = 0         # this is for making fire red
        images[non_bg_idxs] = 0.199 * images[non_bg_idxs] + 0.799 * outputs[non_bg_idxs]
        plt.imsave(os.path.join(ex_path, name[0][0]), images)
        plt.close()