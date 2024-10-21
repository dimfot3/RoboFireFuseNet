import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.tools import *
from utils.pre_training_tools import *
from utils.logger import PreTrainingLogger


def train(model, train_data, args, logger):
    train_loader = DataLoader(train_data, batch_size=args['BATCHSIZE'], shuffle=True, num_workers=args['NUM_WORKERS'], prefetch_factor=2)
    trainer = Trainer(args, model, len(train_data))
    for epoch in range(args['EPOCHS']):
        running_loss = 0
        trainer.optimizer.zero_grad()
        # train loop
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            loss = trainer.training_step(batch)
            running_loss += loss.item() / len(train_loader)
        # metrics
        logger.print_metrics({'loss': running_loss})
        logger.log({'loss': running_loss}, epoch, trainer.scheduler)
        if epoch % args['VALID_FREQ'] == 0:
            model.save_model(os.path.join('weights', args['PROJECTNAME'], args['SESSIONAME']), epoch)
            qualitive_eval_pretrain(lambda data: trainer.inference(data), train_data, 
                   ex_path=f'./outputs/{args["PROJECTNAME"]}/{args["SESSIONAME"]}/visualizations', name=f'Epoch_{epoch + 1}.png')
    print(f"Training Finished! Best Epoch {logger.best_epoch}: MIOU {logger.best_loss}")
    return logger.best_loss

def main(args):
    set_reproducibility(args['SEED'])
    train_dataset = get_dataset(args)
    model = get_model(args)
    logger = PreTrainingLogger(args)
    train(model, train_dataset, args, logger)
    logger.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
