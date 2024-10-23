import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.tools import *
from utils.training_tools import *
from utils.logger import Logger
import shutil


def test(trainer, test_dataset, args, logger):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args['NUM_WORKERS'])
    running_loss, conf_mat, conf_mat_inst = 0.0, np.zeros((args['NUM_CLASSES'], args['NUM_CLASSES'])), np.zeros((args['NUM_CLASSES'], args['NUM_CLASSES']))
    for batch in tqdm(test_loader, desc='Validation'):
        loss, curr_conf_mat, cur_conf_mat_inst = trainer.test_step(batch)
        conf_mat += curr_conf_mat
        conf_mat_inst += cur_conf_mat_inst
        running_loss += loss.item() / len(test_loader)
    metrics = calculate_metrics(conf_mat, running_loss, cls_names=args['CLS_NAMES'], cls_weights=args['CLASS_WEIGHTS'], val=True)
    metrics_inst = calculate_metrics(conf_mat_inst, running_loss, cls_names=args['CLS_NAMES'], cls_weights=args['CLASS_WEIGHTS'], val=True)
    logger.print_metrics(metrics, cls_names=args['CLS_NAMES'], num_classes=args['NUM_CLASSES'], cls_weights=args['CLS_NAMES'], val=True)
    print('-'*shutil.get_terminal_size()[0])
    logger.print_metrics(metrics_inst, cls_names=args['CLS_NAMES'], num_classes=args['NUM_CLASSES'], cls_weights=args['CLS_NAMES'], val=True)
    qualitive_eval(lambda data: trainer.inference(data), test_dataset, 
                   ex_path=f'./outputs/{args["PROJECTNAME"]}/{args["SESSIONAME"]}/visualizations', name=f'TEST.png')
    return metrics

def main(args):
    set_reproducibility(args['SEED'])
    _, _, test_dataset = get_dataset(args, True)
    model = get_model(args)
    args['ONLINELOG'] = False
    logger = Logger(args)
    trainer = Trainer(args, model, len(test_dataset))    
    test(trainer, test_dataset, args, logger)
    logger.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
