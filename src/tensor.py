'''
    Description:
        Contains all tensor related class
    Author:
        Jonver Oro

'''

# TODO: Create backward prop flow for tensors

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

        self.ops='assign'
        self.grad = None
        self.act = activations()
        self.inherit = inherit
        self.requires_grad = requires_grad
        self.weight = None

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

    # * add
    def add(self,tensor,inherit=True):
        '''
            tensor <class>: this is the tensor object to perform operation
            inherit <bool>: set as true if you want to reassign new value to current tensor
        '''
        ops_val = 'add'
        # self.cache.save(self)
        self.cache.combine(tensor.cache)
        val = self.val + tensor.val
        #self.grad = tensor.val-self.val
        return self.__compile(val,inherit,ops_val)
    

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
        prev_grad = np.array(tensor.val[0])
        dx = prev_grad.dot(self.weight.T)
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

        # shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
        # return tensor.val.reshape(shape) + np.zeros_like(input)
        axis = self.weight
        shape = [1 if axis is None or i in axis else self.val.shape[i] for i in range(len(self.val.shape))]
        return tensor.val.reshape(shape) + np.zeros_like(self.val)
        # return tensor.val * np.ones_like(self.val)


    # * activation functions
    def activ(self,funct,inherit=True):

        ops_val = f'act_{funct}'
        if funct == 'sigmoid':
            val = self.act.sigmoid(self.val)
            #self.grad = self.act.d_sigmoid(val)
        elif funct == 'relu':
            val = self.act.relu(self.val)
            #self.grad = self.act.d_relu(val)
        elif funct == 'tanh':
            val = self.act.tanh(self.val)
            #self.grad = self.act.d_tanh(val)

        return self.__compile(val,inherit,ops_val)
        

    def transpose(self):
        self.val = self.val.T
        return self


    def backward(self):
        #TODO: Implement back-propagation 
        prev_grad = None

        # add current state to cache
        # copy_tensor = Tensor(self.val)
        # copy_tensor.ops = self.ops
        # copy_tensor.weight = self.weight
        # self.cache.save(copy_tensor)
        cache_copy = []
        for i,t in enumerate(reversed(self.cache.tensors)):
            print(i,t.ops)
            if i == 0 or prev_grad.shape == (1,):
                # set prev grad as 1 when last tensor in accordance to shape of prev tensor
                # or when prev grad shape is equals (1,)
                prev_grad = Tensor(np.ones(self.cache.tensors[i-1].shape),requires_grad=False,inherit=False) 
                #print(prev_grad)
            else:
                prev_grad = Tensor(prev_grad,requires_grad=False,inherit=False)

            # print(i,'prev_grad',prev_grad)
            if t.ops == 'sum':
                t.grad = t.sum_grad(prev_grad)
                # t.weight = np.ones(t.cache.tensors[i].shape)
            
            if t.ops == 'dot':
                #print(i,prev_grad)
                print(t.ops)
                t.weight,t.grad = t.dot_grad(prev_grad)
                # print(i,t.grad)
                # print(i,t.weight)
            # cache_copy.tensors[i] = t
            # print(t.grad)
            prev_grad = t.grad
            cache_copy.append(t)
            # print(t)
            # if i == 0:
            #     print('recheck')
            #     for yy in self.cache.tensors:
            #         print(yy.weight,yy.ops)
            #print(i,t.grad,t.ops,t.weight)


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
                print(c.shape,c.ops,c.weight)
        print('--- Current State ---')
        print(self.val.shape,self.ops,c.weight)
        




if __name__ == "__main__":

    # * test
    x = Tensor().eye((3,3))
    y = Tensor([[2.0,0,-2.0]])
    z = y.dot(x).sum()
    # print(z)
    # print(z.val)
    #z.history()
    # print(z.history())
    z.backward()
    # for c in z.cache.tensors:
    #     print('w',c.weight)
    #     print('ops',c.ops)
    #     print('c',c)
    #     print('---')
    # z.history()

    # y = Tensor().eye((3,3)).val
    # x = Tensor([[2.0,0,-2.0]]).val
    # w = Tensor().ones((1,3)).val
    # print(x)
    # print(y)

    # z = np.dot(y,x.T)
    # print(z)
    # z = np.dot(z,w)
    # print(z)