# Tensor Nano
This is a open source deep learning library project. Currently In progress.
This is mainly for educational purposes.

### Features
1. Autograd Implementation
2. Implements tensor objects like PyTorch
3. Basic tensor operations - dot,mul,add,mean,sum
4. Basic activation functions - relu,sigmoid,tanh

### Examples

```python
x = Tensor().eye((3,3),requires_grad=True)
y = Tensor([[2.0,0,-2.0]],requires_grad=True)
b = Tensor().eye((1,1),requires_grad=True)

z = y.dot(x).add(b).mean()
print('z',z)
z.backward()

print(b.grad) # dz/db
print(y.grad) # dz/dy
print(x.grad) # dz/dx

```

### PyTorch Example

```python
import torch
x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
b = torch.eye(1, requires_grad=True)
z = y.dot(x).add(b).mean()
z.backward()

print(b.grad)  # dz/db
print(x.grad)  # dz/dx
print(y.grad)  # dz/dy


```

### Tiny Neural Net Example
```python
# * Model class
class TinyNet:

    def __init__(self):
        self.l1 = Tensor().uniform((2,1))
        self.l2 = Tensor().uniform((1,1))

    def forward(self,x):
        a = x.dot(self.l1).relu().dot(self.l2).sigmoid()
        return a

epochs = 1500
batch_size = 32

model = TinyNet()
opt = optim.SGD([model.l1, model.l2], lr=0.0001)
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
```


### TODO
1. Support GPU
2. Support Conv1D, Conv2D, LTSM, RNN, etc operations
3. Improve performance.
4. Fix Bugs
5. Create SOTA model implementations
6. Create performance unittesting


