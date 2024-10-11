import numpy as np


class CustomPolynomialDecayLR:
    def __init__(self, optimizer, base_lr, max_iters, power=0.9, nbb_mult=10):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.power = power
        self.nbb_mult = nbb_mult
        self.cur_iters = 0

    def step(self):
        self.cur_iters += 1
        self._adjust_learning_rate()

    def _adjust_learning_rate(self):
        lr = max(np.real(self.base_lr * ((1 - float(self.cur_iters) / self.max_iters + 1e-5) ** self.power)), 1e-6)
        self.optimizer.param_groups[0]['lr'] = lr
        if len(self.optimizer.param_groups) == 2:
            self.optimizer.param_groups[1]['lr'] = lr * self.nbb_mult
        return lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
