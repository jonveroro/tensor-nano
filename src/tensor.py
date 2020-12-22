'''
    Description:
        Contains all tensor related class
    Author:
        Jonver Oro

'''

# TODO: Create grad tracking per tensor object

import numpy as np
from copy import deepcopy
from activations import activations

class TensorTrack:
    '''
        Serves as cache for tensors
    '''
    def __init__(self):
        self.tensors = []

    def __repr__(self):
        return f"Total {len(self.tensors)} tensors states.\n {self.tensors} "

    def save(self,tensor):
        #self.tensors = tensor.cache.tensors
        self.tensors.append(tensor)

    def combine(self,cache):
        self.tensors = cache.tensors + self.tensors

    

class Tensor:
    '''
        Wraps additional functions to numpy object
    '''
    def __init__(self,val=None,transpose=False,inherit=True,requires_grad=True,**kwargs):
        # self.cache = [] # * stores previous values for derivative computations
        self.cache = TensorTrack()
        if val is not None:
            self.val = np.array(val,**kwargs)
        else:
            self.val = None
        if transpose ==  True:
            self.val = self.val.T

        # * default variables
        self.ops='assign'
        self.grad = None
        self.act = activations()
        self.inherit = inherit
        self.requires_grad = requires_grad
        self.weight = None
        self.weight_grad = None

        # * save initial copy
        # self.cache.save(self)

    def __repr__(self):
        return f"{self.val}"

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
        self.val = np.eye(shape[0],shape[1],**kwargs)
        return self

    def random(self,shape,**kwargs):
        assert(shape[0] >0)
        assert(shape[1]> 0)
        self.val = np.random.rand(shape[0],shape[1],**kwargs)
        return self

    def zeros(self,shape,**kwargs):
        assert(shape[0] >0)
        assert(shape[1]> 0)
        self.val = np.zeros(shape,**kwargs)
        return self

    def ones(self,shape,**kwargs):
        assert(shape[0] >0)
        assert(shape[1]> 0)
        self.val = np.ones(shape,**kwargs)
        return self

    # * TENSOR MAIN OPERATIONS

    def unbroadcast(self,out, in_sh):
        # adjoint operation to broadcast is sum. Need to sum all axis with 1 = in_sh[i] < out.shape[i]
        sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1]) if in_sh != (1,) else None
        return out.sum(axis=sum_axis).reshape(in_sh)

    # * add
    def add(self,tensor,inherit=True):
        '''
            tensor <class>: this is the tensor object to perform operation
            inherit <bool>: set as true if you want to reassign new value to current tensor
        '''
        ops_val = 'add'
        # self.cache.save(self)
        self.cache.combine(tensor.cache)
        self.weight = tensor.val
        val = self.val + tensor.val
        #self.grad = tensor.val-self.val
        return self.__compile(val,inherit,ops_val,weight=tensor.val)

    def add_grad(self,tensor):
        val_shape = self.val.shape
        weight_shape = self.weight.shape
        dx = self.unbroadcast(tensor.val,val_shape)
        dy = self.unbroadcast(tensor.val,weight_shape)
        return dy,dx
    

    # * dot
    def dot(self,tensor,inherit=True):
        '''
            tensor <class>: this is the tensor object to perform operation
            inherit <bool>: set as true if you want to reassign new value to current tensor
        '''
        ops_val = 'dot'
        # self.cache.save(self)
        self.cache.combine(tensor.cache)
        self.weight = tensor.val
        val = self.val.dot(tensor.val)
        return self.__compile(val,inherit,ops_val,weight=tensor.val)

    def dot_grad(self,tensor):
        
        #print(self.val)
        dx = tensor.val.dot(self.weight.T)
        dw = self.val.T.dot(tensor.val)
        
        return dw,dx


    # * sub
    def sub(self,tensor,inherit=True):
        '''
            tensor <class>: this is the tensor object to perform operation
            inherit <bool>: set as true if you want to reassign new value to current tensor
        '''
        ops_val = 'sub'
        self.cache.save(self)
        self.cache.combine(tensor.cache)
        val = self.val - tensor.val
        #self.grad = tensor.val-self.val
        return self.__compile(val,inherit,ops_val)


    # * sum
    def sum(self,axis=None):
        ops_val = 'sum'
        #self.cache.save(self)
        # print(self.cache)
        val = np.array([self.val.sum()]) if axis is None else self.val.sum(axis=axis)
        self.weight = axis
        return self.__compile(val,self.inherit,ops_val)

    def sum_grad(self,tensor):

        axis = self.weight
        shape = [1 if axis is None or i in axis else self.val.shape[i] for i in range(len(self.val.shape))]
        return tensor.val.reshape(shape) + np.zeros_like(self.val)


    #  * activation operations
    def sigmoid(self):
        ops_val = 'sigmoid'
        val = self.act.sigmoid(self.val)
        return self.__compile(val,self.inherit,ops_val)

    def sigmoid_grad(self):
        return self.act.d_sigmoid(self.val)

    def relu(self):
        ops_val = 'relu'
        val = self.act.relu(self.val)
        return self.__compile(val,self.inherit,ops_val)
    
    def relu_grad(self):
        return self.act.d_relu(self.val)

    def tanh(self):
        ops_val = 'tanh'
        val = self.act.tanh(self.val)
        return self.__compile(val,self.inherit,ops_val)

    def tanh_grad(self):
        return self.act.d_tanh(self.val)

    def transpose(self):
        self.val = self.val.T
        return self


    def backward(self):
        #TODO: Implement back-propagation 
        prev_grad = None
        cache_copy = []
        for i,t in enumerate(reversed(self.cache.tensors)):
            if i == 0 or prev_grad.shape == (1,):
                # set prev grad as 1 when last tensor in accordance to shape of prev tensor
                # or when prev grad shape is equals (1,)
                prev_grad = Tensor(np.ones(self.cache.tensors[i-1].shape),requires_grad=False,inherit=False) 
            else:
                prev_grad = Tensor(prev_grad,requires_grad=False,inherit=False)

            # * matix ops
            if t.ops == 'sum':
                t.grad = t.sum_grad(prev_grad)
            
            if t.ops == 'dot':
                t.weight_grad,t.grad = t.dot_grad(prev_grad)

            if t.ops == 'add':
                t.weight_grad,t.grad = t.add_grad(prev_grad)

            # * activation ops
            if t.ops == 'relu':
                t.grad = t.relu_grad()
                
            if t.ops == 'sigmoid':
                t.grad = t.sigmoid_grad()

            if t.ops == 'tanh':
                t.grad = t.tanh_grad()

            prev_grad = t.grad
            cache_copy.append(t)

        self.cache.tensors = cache_copy


        return self

    # * others
    def __compile(self,val,inherit,ops,weight=None):
        # * compiles return value of tensor
        if self.inherit ==False:
            return Tensor(val)
        else:
            # * copy current tensor to a new tensor object
            copy_tensor = Tensor(val)
            copy_tensor.ops = ops
            copy_tensor.weight = self.weight
            self.cache.save(copy_tensor) # * save a copy of itself
            copy_tensor.cache = self.cache

            # * replace old values
            self.val = val
            self.ops = ops
            
            return self

    def get_prev(self,step=1):
        # return previous state of tensor
        try:
            assert(len(self.cache.tensors)>0)
            return self.cache.tensors[len(self.cache.tensors)-step]
        except AssertionError as e:
            print('Error: Tensor has no previous states')
            raise(e)

    def history(self):
        # * return tensor state changes
        if len(self.cache.tensors) > 0:
            print('-- Previous States ---')
            for c in self.cache.tensors[:len(self.cache.tensors)]:
                print(c.shape,c.ops)
        print('--- Current State ---')
        print(self.val.shape,self.ops,c.weight)
        




if __name__ == "__main__":

    # * test
    b = Tensor().eye((1,3))
    x = Tensor().eye((3,3))
    y = Tensor([[2.0,0,-2.0]])
    z = y.dot(x).add(b).sum()
    print('z =',z)
    z.backward() # * backprop
    
    # print derivatives
    print('derivative values')
    for c in z.cache.tensors:
        print(f'--{c.ops}---')
        print(c.grad) # dz/dy 
        print(c.weight_grad) # dz/dx


