'''
    Description:
        Contains all tensor related class
    Author:
        Jonver Oro
'''

import numpy as np

class Tensor:
    '''
        Wraps additional functions to numpy library
    '''
    def __init__(self,val=None,transpose=False,**kwargs):
        self.cache = [] # * stores previous values for derivative computations
        if val is not None:
            self.val = np.array(val,**kwargs)
        else:
            self.val = None
        if transpose ==  True:
            self.val = self.val.T

        self.ops=None
    
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

    # * tensor operation functions
    def add(self,tensor,inherit=True):
        '''
            tensor <class>: this is the tensor object to perform operation
            inherit <bool>: set as true if you want to reassign new value to current tensor
        '''
        val = self.val + tensor.val
        return self.__compile(val,inherit)
    
    def multiply(self,tensor,inherit=True):
        '''
            tensor <class>: this is the tensor object to perform operation
            inherit <bool>: set as true if you want to reassign new value to current tensor
        '''
        val = np.dot(self.val,tensor.val)
        return self.__compile(val,inherit)

    def subtract(self,tensor,inherit=True):
        '''
            tensor <class>: this is the tensor object to perform operation
            inherit <bool>: set as true if you want to reassign new value to current tensor
        '''
        val = self.val +  tensor.val
        return self.__compile(val,inherit)

    
    # * others
    def __compile(self,val,inherit):
        # * compiles return value of tensor
        if inherit ==False:
            return Tensor(val)
        else:
            self.cache.append(Tensor(self.val)) # * saves a copy of itself
            self.val = val
            return self

    def get_prev(self,step=1):
        # return previous state of tensor
        try:
            assert(len(self.cache)>0)
            return self.cache[len(self.cache)-step]
        except AssertionError as e:
            print('Error: Tensor has no previous states')
            raise(e)
    





if __name__ == "__main__":
    w = Tensor().random((2,2))
    x = Tensor().random((2,2))
    y = Tensor().random((2,2))
    A = x.multiply(y).add(w)
    for c in A.cache:
        print(c.val)
    print(A.val)

