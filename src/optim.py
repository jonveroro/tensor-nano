
from copy import deepcopy

class SGD:
    def __init__(self,params,lr=0.0001):
        self.params = params
        self.lr = lr

    def step(self):
        n_params = self.params
        for e,t in enumerate(n_params):
            t.val -= t.grad * self.lr
            n_params[e] = t

        self.params = deepcopy(n_params)
