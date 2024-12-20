import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.tools import *
from utils.training_tools import *
from utils.logger import Logger


def train(model, train_data, val_data, args, logger):
    train_loader = DataLoader(train_data, batch_size=args['BATCHSIZE'], shuffle=True, num_workers=args['NUM_WORKERS'], prefetch_factor=1, drop_last=True,
    pin_memory=True)
    trainer = Trainer(args, model, len(train_data))
    for epoch in range(trainer.start_epoch, args['EPOCHS']):
        running_loss, conf_mat = 0, np.zeros((args['NUM_CLASSES'], args['NUM_CLASSES']))
        # train loop
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            loss, curr_conf_mat = trainer.training_step(batch)
            conf_mat += curr_conf_mat
            running_loss += loss.item() / len(train_loader)
        metrics, _ = calculate_metrics(conf_mat, running_loss, cls_names=args['CLS_NAMES'], cls_weights=args['CLASS_WEIGHTS'])
        # validation, saving model and examining stop criterion
        if epoch % args['VALID_FREQ'] == 0:
            val_metrics, iou = valid(trainer, val_data, epoch, args)
            logger.print_metrics(val_metrics, args['CLS_NAMES'], args['NUM_CLASSES'], cls_weights=args['CLS_NAMES'], val=True)
            metrics = {**metrics, **val_metrics}
            trainer.save_checkpoint(os.path.join('weights', args['PROJECTNAME'], args['SESSIONAME']), epoch)
            if trainer.stop_sign(metrics): break
            train_data.mious = iou
        logger.log(metrics, epoch, trainer.scheduler)
        logger.print_metrics(metrics, cls_names=args['CLS_NAMES'], num_classes=args['NUM_CLASSES'], cls_weights=args['CLS_NAMES'], val=False)
    print(f"Training Finished! Best Epoch {logger.best_epoch}: MIOU {logger.best_miou}")
    return logger.best_miou

def valid(trainer, val_data, cur_epoch, args):
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=args['NUM_WORKERS'])
    running_loss, conf_mat = 0.0, np.zeros((args['NUM_CLASSES'], args['NUM_CLASSES']))
    for batch in tqdm(valid_loader, desc='Validation'):
        loss, curr_conf_mat = trainer.valid_step(batch)
        conf_mat += curr_conf_mat
        running_loss += loss.item() / len(valid_loader)
    metrics, iou = calculate_metrics(conf_mat, running_loss, cls_names=args['CLS_NAMES'], cls_weights=args['CLASS_WEIGHTS'], val=True)
    qualitive_eval(lambda data: trainer.inference(data), val_data, 
                   ex_path=f'./outputs/{args["PROJECTNAME"]}/{args["SESSIONAME"]}/visualizations', name=f'Epoch_{cur_epoch}.png')
    return metrics, iou

def main(args):
    set_reproducibility(args['SEED'])
    train_dataset, val_dataset, _ = get_dataset(args)
    model = get_model(args)
    logger = Logger(args)
    train(model, train_dataset, val_dataset, args, logger)
    logger.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
