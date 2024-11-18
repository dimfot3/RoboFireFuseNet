import numpy as np
import math


class PolynomialDecayLR:
    def __init__(self, optimizer, base_lr, max_iters, num_batches, power=0.9, nbb_mult=10, warmup_epochs=0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.power = power
        self.nbb_mult = nbb_mult
        self.cur_iters = 0
        self.cur_epoch = 0
        self.num_batches = num_batches
        self.warmup_epochs = warmup_epochs

    def step(self):
        self.cur_iters += 1 if self.cur_epoch >= self.warmup_epochs else 0
        self.cur_epoch += 1 / self.num_batches
        self._adjust_learning_rate()

    def _adjust_learning_rate(self):
        if self.cur_epoch < self.warmup_epochs:
            lr = self.base_lr * self.cur_epoch / self.warmup_epochs
        else:
            lr = max(np.real(self.base_lr * (1 - float(self.cur_iters) / self.max_iters + 1e-12) ** self.power), 1e-8)
        self.optimizer.param_groups[0]['lr'] = lr
        if len(self.optimizer.param_groups) == 2:
            self.optimizer.param_groups[1]['lr'] = lr * self.nbb_mult
        return lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {
            'cur_iters': self.cur_iters,
            'cur_epoch': self.cur_epoch
        }

    def load_state_dict(self, state_dict):
        self.cur_iters = state_dict['cur_iters']
        self.cur_epoch = state_dict['cur_epoch']

class CosineDecay:
    def __init__(self, optimizer, base_lr, epochs, num_batches, warmup):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_epochs = epochs
        self.num_batches = num_batches
        self.cur_epoch = 0
        self.warmup_epochs = warmup
        self.nbb_mult = 10

    def step(self):
        self.cur_epoch += 1 / self.num_batches
        self._adjust_learning_rate()

    def _adjust_learning_rate(self):
        if self.cur_epoch < self.warmup_epochs:
            lr = self.base_lr * self.cur_epoch / self.warmup_epochs
        else:
            lr = self.base_lr  * 0.5 * \
            (1. + math.cos(math.pi * (self.cur_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
        self.optimizer.param_groups[0]['lr'] = lr
        if len(self.optimizer.param_groups) == 2:
            self.optimizer.param_groups[1]['lr'] = lr # * self.nbb_mult
        return lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {
            'cur_epoch': self.cur_epoch
        }

    def load_state_dict(self, state_dict):
        self.cur_epoch = state_dict['cur_epoch']
