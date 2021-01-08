'''
    Description:
        Contains all tensor related class. Performs autograd functionality for every tensors

    References:
        https://github.com/geohot/tinygrad/blob/master/tinygrad/ops_cpu.py
        https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py

'''

# TODO: Add tensor movement ops
import time
import numpy as np
from copy import deepcopy,copy
from activations import activations

class Ops:
    '''
        Contains all tensor operations
    '''
    def __init__(self):
        self._backward = lambda: None

    # * basic ops

    def sub(self,weight):
        out =Tensor(self.val-weight.val,(self,weight),'sub')
        self.weight = weight

        def __backward():
            self.grad = self.unbroadcast(out.grad,self.val.shape)
            weight.grad = self.unbroadcast(-out.grad,self.weight.shape)
            assert(self.grad.shape == self.val.shape)

        out._backward = __backward
        return out
            

    def dot(self,weight):

        out = Tensor(self.val.dot(weight.val), (self, weight), 'dot')
        self.weight = weight

        # * backward function
        def __backward():
            self.grad = out.grad.dot(self.weight.val.T)
            weight.grad = self.val.T.dot(out.grad)
            assert(self.grad.shape == self.val.shape)
        
        out._backward = __backward

        return out


    def mul(self, weight):
        
        out = Tensor(self.val * weight.val, (self, weight), 'mul')
        self.weight = weight

        # * backward function
        def __backward():
            self.grad = weight.val * out.grad
            weight.grad = self.val * out.grad

        out._backward = __backward
        return out

    def add(self,weight):

        out = Tensor(self.val + weight.val, (self,weight),'add')
        self.weight = weight

        # * backward function
        def __backward():
            self.grad = self.unbroadcast(out.grad,self.val.shape)
            weight.grad = self.unbroadcast(out.grad,self.weight.shape)
            assert(self.grad.shape == self.val.shape)

        out._backward = __backward
        return out

    

    def sum(self,axis=None):
        val=  np.array([self.val.sum()]) if axis is None else self.val.sum(axis=axis)
        out = Tensor(val,(self,deepcopy(self)),'sum')
        self.axis = axis

        # * backward function
        def __backward():
            axis = [self.axis] if type(self.axis) is int else self.axis
            shape = [1 if axis is None or i in axis else out.shape[i] for i in range(len(out.shape))]
            self.grad = out.grad.reshape(shape) + np.zeros_like(self.val)
            assert(self.grad.shape == self.val.shape)
            
        out._backward = __backward
        return out

    def mean(self,axis=None):
        #val = self.val.sum(axis=axis)
        val = self.val.mean()
        out = Tensor(np.array([val]),(self,deepcopy(self)),'mean')
        
        # * backward function
        def __backward():
            axis = [self.axis] if type(self.axis) is int else self.axis
            shape = [1 if axis is None or i in axis else out.shape[i] for i in range(len(out.shape))]
            self.grad = (out.grad.reshape(shape) + np.zeros_like(self.val)) * ((np.prod(out.shape)/np.prod(self.shape)))
            assert(self.grad.shape == self.val.shape)
            
        out._backward = __backward
        return out
        


    # * activations
    def relu(self):
        out = Tensor(activations().relu(self.val),(self,deepcopy(self)),'relu')
        # * backward
        def __backward():
            self.grad = activations().d_relu(out.val)

        out._backward = __backward
        return out

    
    def sigmoid(self):
        out = Tensor(activations().sigmoid(self.val),(self,deepcopy(self)),'sigmoid')
        # * backward
        def __backward():
            self.grad = activations().d_sigmoid(out.val)
            assert(self.grad.shape == self.val.shape)

        out._backward = __backward
        return out

    def tanh(self):
        out = Tensor(activations().tanh(self.val),(self,deepcopy(self)),'tanh')
        # * backward
        def __backward():
            self.grad = activations().d_tanh(out.val)
            assert(self.grad.shape == self.val.shape)

        out._backward = __backward
        return out

    # * etc
    def unbroadcast(self,out, in_sh):
        # adjoint operation to broadcast is sum. Need to sum all axis with 1 = in_sh[i] < out.shape[i]
        sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1]) if in_sh != (1,) else None
        return out.sum(axis=sum_axis).reshape(in_sh)
        

class Tensor(Ops):
    '''
        Wraps additional functions to numpy object
    '''
    def __init__(self,val=None,_cache=(), _op='init',requires_grad=True,**kwargs):
        super().__init__()
        if val is not None:
            self.val = np.array(val,dtype=np.float32,**kwargs)
        else:
            self.val = None

        self.grad = 0
        self.weight = None
        
        self._prev = set(_cache)
        self.ops = _op 
        self._sequence = []
        self.axis = None
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Tensor({self.val})"

    @property
    def shape(self):
        return self.val.shape

    @property
    def dtype(self):
        return self.val.dtype


    # * tensor default value generation function
    def eye(self,shape,**kwargs):
        assert(shape[0] >0)
        assert(shape[1]> 0)
        self.val = np.eye(shape[0],shape[1],**kwargs).astype(np.float32)
        return self

    def random(self,shape,**kwargs):
        assert(shape[0] >0)
        assert(shape[1]> 0)
        self.val = np.random.rand(shape[0],shape[1],**kwargs).astype(np.float32)
        return self

    def zeros(self,shape,**kwargs):
        assert(shape[0] >0)
        assert(shape[1]> 0)
        self.val = np.zeros(shape,**kwargs).astype(np.float32)
        return self

    def ones(self,shape,**kwargs):
        assert(shape[0] >0)
        assert(shape[1]> 0)
        self.val = np.ones(shape,**kwargs).astype(np.float32)
        return self

    def uniform(self,shape,**kwargs):
        assert(shape[0] >0)
        assert(shape[1]> 0)
        self.val = np.random.uniform(-1., 1., size=shape)/np.sqrt(np.prod(shape)).astype(np.float32)
        return self


    def backward(self):
        assert(self.shape == (1,))
        # * get tensor traversal sequence
        sequence = []
        traversed = []
        # * tensor traversal function
        def __get_traversal(tensor):
            if tensor not in traversed:
                traversed.append(tensor)
                for c in tensor._prev:
                    # * process child tensors
                    __get_traversal(c)
                # * append current tensor
                sequence.append(tensor)
        # * process itself
        __get_traversal(self)
        # * apply backward steps for tensors
        self._sequence = sequence
        # * create default grad val in shape of current value
        self.grad =  np.ones(self.val.shape,dtype=self.val.dtype)
        # * backward prop loop
        for t in reversed(sequence):
            if t.ops != 'init':
                if t.requires_grad == True:
                    assert (t.grad is not None)
                    t._backward()
    

    def history(self):
        print('----Tensor state history---')
        for t in reversed(self._sequence):
            print(t.shape,t.ops)
            


if __name__ == "__main__":

    start_time = time.time()
    x = Tensor().eye((3,3))
    y = Tensor([[2.0,0,-2.0]])
    b = Tensor().eye((1,1))

    z = y.dot(x).add(b).mean()
    print('z',z)
    z.backward()
    print("--- %s seconds ---" % (time.time() - start_time))
    print('dz',z.grad)
    print('db',b.grad)
    print('dy',y.grad)
    print('dx',x.grad)
    #z.history()
    
    

