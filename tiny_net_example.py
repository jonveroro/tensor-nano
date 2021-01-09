'''
    This is a simple example on how to train a basic neural network
'''

from sklearn import datasets
import numpy as np
from inspect import signature
from copy import deepcopy
from src.tensor import Tensor
import src.optim as optim


# * test data
X, Y = datasets.make_moons(1000, noise=0.05)
print(X.shape)
print(Y.shape)

# * Model class
class TinyNet:

    def __init__(self):
        self.l1 = Tensor().uniform((2,1))
        self.l2 = Tensor().uniform((1,1))

    def forward(self,x):
        a = x.dot(self.l1).relu().dot(self.l2).sigmoid()
        return a

epochs = 1000
batch_size = 32

model = TinyNet()
opt = optim.Adam([model.l1, model.l2],lr=0.0001)
verbose = 1

print('Traning simple neural net')
for e in range(0,epochs):

    loss_list = []

    for b in range(0,len(X),batch_size):
        # * batch training
        x = X[b:b+batch_size]
        y = Y[b:b+batch_size]
        y = y.reshape(y.shape[0],1)

        x = Tensor(x)
        y = Tensor(y)

        out = model.forward(x)
        loss = out.mul(y).mean()
        loss.backward()
        opt.step()

        # TODO: improve weight update method
        # * deep copy new weight values
        model.l1 = deepcopy(opt.params[0])
        model.l2 = deepcopy(opt.params[1])

        #print(loss.val[0])
        loss_list.append(loss.val[0])

    # * display loss value
    loss_main = np.mean(np.array(loss_list))
    if e % verbose == 0:
        print(f'\rEpoch: {e} Loss: {loss_main}', end='')

print('\nDone Training')
