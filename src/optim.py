
# TODO: Create Other Optimizers

from copy import deepcopy

class SGD:
    def __init__(self,params,lr=0.0001):
        self.params = params
        self.lr = lr

    def step(self):
        for t in self.params:
            t.val -= t.grad * self.lr

