
# TODO: Create Other Optimizers

from copy import deepcopy
import numpy as np

class SGD:
    def __init__(self,params,lr=0.001):
        self.params = params
        self.lr = lr

    def step(self):
        for t in self.params:
            t.val -= t.grad * self.lr


class Adam:

    def __init__(self,params,lr=0.0001,b1=0.9,b2=0.999,e=1e-8):
        self.params = params
        self.lr = lr # learning rate
        self.b1 = b1 # beta 1
        self.b2 = b2 # beta 2
        self.e = e # epsilon

        # * create momentum init values
        self.v = []
        self.s = []
        self.t = 0
        for p in self.params:
            n = np.zeros(p.shape).astype(np.float32)
            self.v.append(n)
            self.s.append(n)


    def step(self):
        self.t +=1
        for e,p in enumerate(self.params):
            self.v[e] = self.b1 * self.v[e] + (1.0-self.b1) * p.grad # vdw
            self.s[e] = self.b2 * self.s[e] +((1.0-self.b2)*(p.grad*p.grad)) #sdw
            vc = self.v[e]/(1.0-self.b1**self.t) # v corrected
            sc = self.s[e]/(1.0-self.b2**self.t) # s corrected
            p.val -= self.lr * (vc/(np.sqrt(sc)+self.e)) # update values
            #print(p.val)
