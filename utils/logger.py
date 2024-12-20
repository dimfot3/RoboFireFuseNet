import wandb
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import shutil


class WDBLogger:
    def __init__(self, args, model=None, sync_tensorboard=False):
        # if sync_tensorboard:
        #     log_dir = os.path.join('outputs', args['PROJECTNAME'], args['SESSIONAME'], 'logs')
        wandb.init(config=args, project=args['PROJECTNAME'], name=args['SESSIONAME'])
        if model != None:
            wandb.watch(model, log='all')
        
    def log(self, metrics, epoch, scheduler):
        for i, lr in enumerate(np.array(scheduler.get_last_lr()).reshape(-1, )):
            metrics[f'LR {i}'] = lr
        wandb.log(metrics, epoch)
    
    def close(self):
        wandb.finish()
        # shutil.rmtree('./wandb')

class TensorBoardLogger:
    def __init__(self, args, model=None):
        self.log_dir = os.path.join('outputs', args['PROJECTNAME'], args['SESSIONAME'], 'logs')
        self.writer = SummaryWriter(self.log_dir)
        args_text = '\n'.join([f'{key}: {value}' for key, value in args.items()])
        self.writer.add_text("config", args_text, 0)
        if model is not None:
            pass
            dummy_input = torch.randn(1, *args['input_shape']).to(next(model.parameters()).device)
            self.writer.add_graph(model, dummy_input)
    
    def log(self, metrics, epoch, scheduler):
        for i, lr in enumerate(np.array(scheduler.get_last_lr())):
            metrics[f'LR {i}'] = lr
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)
        self.writer.flush()

    def close(self):
        self.writer.close()

class Logger:
    def __init__(self, args, model=None):
        self.best_miou, self.best_epoch = 0, 0        # this is a logger metric
        self.online_log = args['ONLINELOG']
        if self.online_log:
            self.wdb_log = WDBLogger(args, model, sync_tensorboard=True)
        self.tensor_log = TensorBoardLogger(args, model)
    
    def log(self, metrics, epoch, scheduler):
        # this logs best metric
        if 'val miou' in list(metrics.keys()):
            if metrics['val miou'] > self.best_miou:
                self.best_miou, self.best_epoch = metrics['val miou'], epoch
        metrics['global_step'] = epoch
        metrics['best MIOU'] = self.best_miou
        self.tensor_log.log(metrics, epoch, scheduler)
        if self.online_log:
            self.wdb_log.log(metrics, epoch, scheduler)

    def close(self):
        self.tensor_log.close()
        if self.online_log:
            self.wdb_log.close()
    
    def add_hyperparams(self, params, metrics):
        self.tensor_log.writer.add_hparams(params, metrics, run_name='./')

    def print_metrics(self, metrics, cls_names=None, num_classes=0, cls_weights=None, val=False):
        avg_f1 = metrics.get(f'{"val " if val else ""}avg_f1', 0)
        avg_acc = metrics.get(f'{"val " if val else ""}avg_acc', 0)
        avg_precision = metrics.get(f'{"val " if val else ""}avg_precision', 0)
        avg_recall = metrics.get(f'{"val " if val else ""}avg_recall', 0)
        total_acc = metrics.get(f'{"val " if val else ""}total_acc', 0)
        miou = metrics.get(f'{"val " if val else ""}miou', 0)
        avgloss = metrics.get(f'{"val " if val else ""}avgloss', 0)
        names = np.arange(num_classes) if cls_names is None else cls_names
        cls_weights = np.ones(num_classes) if cls_weights is None else cls_weights
        
        precision = [metrics.get(f'{"val " if val else ""}precision {name}', 0) for name in names]
        recall = [metrics.get(f'{"val " if val else ""}recall {name}', 0) for name in names]
        f1_score = [metrics.get(f'{"val " if val else ""}f1_score {name}', 0) for name in names]
        accuracy = [metrics.get(f'{"val " if val else ""}accuracy {name}', 0) for name in names]
        iou = [metrics.get(f'{"val " if val else ""}iou {name}', 0) for name in names]
        weighted_f1 = metrics.get(f'{"val " if val else ""}weighted_f1', 0)

        print("Average Metrics:")
        print(f" - Average Loss: {avgloss:.4f}")
        print(f" - Average F1 Score: {avg_f1:.4f}")
        print(f" - Weighted F1: {weighted_f1:.4f}")
        print(f" - MIoU: {miou:.4f}")
        print(f" - Average Precision: {avg_precision:.4f}")
        print(f" - Average Recall: {avg_recall:.4f}")
        print(f" - Average Accuracy: {avg_acc:.4f}")
        print(f" - Total Accuracy: {total_acc:.4f}")
        
        print("\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'IoU Score':<10} {'Accuracy':<10}")
        print("-" * 45)
        
        for i in range(num_classes):
            print(f"{names[i]:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1_score[i]:<10.4f} {iou[i]:<10.4f} {accuracy[i]:<10.4f}")
        
        print(f"Best Epoch {self.best_epoch}: MIOU {self.best_miou}\n")

