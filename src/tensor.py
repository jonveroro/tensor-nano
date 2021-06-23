'''
    Description:
        Contains all tensor related class. Performs autograd functionality for every tensors

    References:
        https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py

'''

# TODO: Add softmax,logsoftmax

import time
import numpy as np
from copy import deepcopy,copy

class OpsNumpy:
    '''
        Contains all lower basic numpy operations with equivalent derivative functions;
        Call functions here for numpy matrix operations
    '''
    @staticmethod
    def dot_np(x,y):
        # * dot function
        out = x.dot(y)
        return out

    @staticmethod
    def dot_np_grad(p_grad,x,y):
        # * dot derivative
        dx= p_grad.dot(y.T)
        dy = x.T.dot(p_grad)
        return dx,dy

    @staticmethod
    def sub_np(x,y):
        # * subtract function
        out = x-y
        return out
    
    @staticmethod
    def sub_np_grad(p_grad,x_shape,y_shape):
        # subtract derivative
        def __unbroadcast(out, in_sh):
            sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1]) if in_sh != (1,) else None
            return out.sum(axis=sum_axis).reshape(in_sh)

        dx = __unbroadcast(p_grad,x_shape)
        dy = __unbroadcast(-p_grad,y_shape)
        return dx,dy

    @staticmethod
    def add_np(x,y):
        # add function
        out = x+y
        return out

    @staticmethod
    def add_np_grad(p_grad,x_shape,y_shape):
        # add derivative
        def __unbroadcast(out, in_sh):
            sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1]) if in_sh != (1,) else None
            return out.sum(axis=sum_axis).reshape(in_sh)

        dx = __unbroadcast(p_grad,x_shape)
        dy = __unbroadcast(p_grad,y_shape)
        return dx,dy

    @staticmethod
    def mul_np(x,y):
        # multiply function
        out = x*y
        return out

    @staticmethod
    def mul_np_grad(p_grad,x,y):
        # * multiply derivative
        def __unbroadcast(out, in_sh):
            sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1]) if in_sh != (1,) else None
            return out.sum(axis=sum_axis).reshape(in_sh)
        dx = __unbroadcast(y*p_grad, x.shape)
        dy = __unbroadcast(x*p_grad, y.shape)

        return dx,dy

    @staticmethod
    def div_np(x,y):
        # * div
        out = x / y
        return out

    @staticmethod
    def div_np_grad(p_grad,x,y):
        # * div derivative
        dx = y / p_grad
        dy = x / p_grad
        return dx,dy

    @staticmethod
    def reshape_np(x,shape):
        # * reshape function
        out = x.reshape(shape)
        return out

    @staticmethod
    def reshape_np_grad(p_grad,x):
        # * reshape derivative
        grad = p_grad.reshape(x.shape)
        return grad

    @staticmethod
    def sum_np(x,axis=None):
        # * sum function
        out = np.array([x.sum()]) if axis is None else x.sum(axis=axis)
        return out

    @staticmethod
    def sum_np_grad(p_grad,out,inp,axis=None):
        # * sum derivative
        axis = [axis] if type(axis) is int else axis
        shape = [1 if axis is None or i in axis else out.shape[i] for i in range(len(out.shape))]
        grad = p_grad.reshape(shape) + np.zeros_like(inp)
        return grad


    @staticmethod
    def max_np(x,axis=None):
        # * max function
        axis = [axis] if type(axis) == int else axis
        out = np.amax(x, axis=None if axis is None else tuple(axis), keepdims=True)
        if axis is not None:
            out = out.reshape([x.shape[i] for i in range(len(x.shape)) if i not in axis])
        return out

    @staticmethod
    def max_np_grad(p_grad,inp,axis,out):
        # * max derivative function
        axis = [axis] if type(axis) == int else axis
        shape = [1 if axis is None or i in axis else inp.shape[i] for i in range(len(inp.shape))]
        out_2 = (inp==out.reshape(shape))
        div = out_2.sum(axis=None if axis is None else tuple(axis), keepdims=True)
        grad = out_2*p_grad.reshape(shape)/div
        return grad

    @staticmethod
    def log_np(x):
        # * log function
        out = np.log(x)
        return out

    @staticmethod
    def log_np_grad(p_grad,x):
        # * log derivative
        grad = p_grad / x
        return grad

    @staticmethod
    def exp_np(x):
        # * exp function
        out = np.exp(x)
        return out

    @staticmethod
    def exp_np_grad(p_grad,out):
        # * exp derivative
        grad = p_grad * out
        return grad


    @staticmethod
    def transpose_np(x,order):
        # * transpose function
        out = np.transpose(x,order)
        return out

    @staticmethod
    def transpose_np_grad(x,p_order):
        # * transpose backward
        grad = np.transpose(x, np.argsort(p_order))
        return grad

    @staticmethod
    def pow_np(x,y):
        # * pow function
        out = x ** y
        return out

    @staticmethod
    def pow_np_grad(p_grad,x,y):
        # pow backward
        def __unbroadcast(out, in_sh):
            sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1]) if in_sh != (1,) else None
            return out.sum(axis=sum_axis).reshape(in_sh)

        x_grad =  __unbroadcast(y * (x**(y-1.0)) * p_grad, x.shape)
        y_grad = __unbroadcast((x**y) * np.log(x) * p_grad, y.shape)

        return x_grad,y_grad
        
        

class OpsMain(OpsNumpy):
    '''
        Contains all main tensor operations needed; 
        Call functions here for applying tensor operations
    '''
    def __init__(self):
        super().__init__()
        self._backward = lambda: None

    # * basic ops
    def sub(self,weight):
        val = self.sub_np(self.val,weight.val)
        out =Tensor(val,(self,weight),'sub')
        self.weight = weight

        def __backward():
            self.grad,weight.grad = self.sub_np_grad(out.grad,self.val.shape,self.weight.shape)
            assert(self.grad.shape == self.val.shape)

        out._backward = __backward
        return out
            

    def dot(self,weight):
        val = self.dot_np(self.val,weight.val)
        out = Tensor(val, (self, weight), 'dot')
        self.weight = weight

        # * backward function
        def __backward():

            self.grad,weight.grad = self.dot_np_grad(out.grad,self.val,self.weight.val)
            assert(self.grad.shape == self.val.shape)
        
        out._backward = __backward

        return out


    def mul(self, weight):
        val = self.mul_np(self.val,weight.val)
        out = Tensor(val, (self, weight), 'mul')
        self.weight = weight

        # * backward function
        def __backward():
            self.grad,weight.grad = self.mul_np_grad(out.grad,self.val,self.weight.val)

        out._backward = __backward
        return out

    def add(self,weight):
        val = self.add_np(self.val,weight.val)
        out = Tensor(val, (self,weight),'add')
        self.weight = weight

        # * backward function
        def __backward():

            self.grad,weight.grad = self.add_np_grad(out.grad,self.val.shape,self.weight.shape)
            assert(self.grad.shape == self.val.shape)

        out._backward = __backward
        return out

    

    def sum(self,axis=None):
        val=  self.sum_np(self.val)
        out = Tensor(val,(self,deepcopy(self)),'sum')
        self.axis = axis

        # * backward function
        def __backward():

            self.grad = self.sum_np_grad(out.grad,out.val,self.val)
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
        out = Tensor(np.maximum(0, self.val),(self,deepcopy(self)),'relu')
        # * backward
        def __backward():
            self.grad = np.where(out.val <= 0, 0, 1)

        out._backward = __backward
        return out

    
    def sigmoid(self):
        val = 1 / (1 + np.exp(-self.val))
        out = Tensor(val,(self,deepcopy(self)),'sigmoid')
        # * backward
        def __backward():
            self.grad = out.val * (1 - out.val)
            assert(self.grad.shape == self.val.shape)

        out._backward = __backward
        return out

    def tanh(self):
        t=(np.exp(self.val)-np.exp(-self.val))/(np.exp(self.val)+np.exp(-self.val))
        out = Tensor(t,(self,deepcopy(self)),'tanh')
        # * backward
        def __backward():
            self.grad = 1-out.val**2
            assert(self.grad.shape == self.val.shape)

        out._backward = __backward
        return out

    def softmax(self):
        n_s = list(self.shape)[:-1]+[1]
        m = self.max_np(self.val,axis=len(self.shape)-1)
        #print('MAX FORWARD',m)
        mr = self.reshape_np(m,n_s)
        #print('RESHAPE FORWARD',mr)
        e_sub = self.sub_np(self.val,mr)
        #print('SUB FORWARD',e_sub)
        e_sub_exp = self.exp_np(e_sub)
        #print('EXP FORWARD',e_sub_exp)
        e_sum = self.sum_np(e_sub_exp,axis=len(self.shape)-1)
        #print('SUM FORWARD',e_sum)
        ss = self.reshape_np(e_sum,n_s)
        #print('RESHAPE FORWARD',ss)
        ss_pow = self.pow_np(ss,-1.0)
        #print('POW FORWARD',ss_pow)
        ss_mul = self.mul_np(ss_pow,e_sub_exp)
        #print('MUL FORWARD',ss_mul)
        out = Tensor(ss_mul,(self,deepcopy(self)),'softmax')

        def __backward():
            # * backward function
            mul_grad_x,mul_grad_y = self.mul_np_grad(out.grad,e_sub_exp,ss_pow)
            #print('MUL BACKWARD',mul_grad_x,mul_grad_y)
            pow_grad_x,pow_grad_y = self.pow_np_grad(mul_grad_y,ss,np.array([-1]))
            #print('POW BACKWARD',pow_grad_x,pow_grad_y)
            ss_grad = self.reshape_np_grad(pow_grad_x,e_sum)
            print(e_sub_exp.shape)
            e_sum_grad = self.sum_np_grad(ss_grad,e_sum,e_sub_exp,axis=len(self.shape)-1)
            #print('SUM BACKWARD',e_sum_grad)
            e_sub_exp_grad = self.exp_np_grad(e_sum_grad,e_sub_exp)
            #print('EXP BACKWARD',e_sub_exp_grad)
            e_sub_grad_x,e_sub_grad_y = self.sub_np_grad(e_sub_exp_grad,self.val.shape,mr.shape)
            #print('SUB BACKWARD', e_sub_grad_x,e_sub_grad_y)
            mr_grad = self.reshape_np_grad(e_sub_grad_y,m)
            #print('RESHAPE BACKWARD',mr_grad)
            m_grad = self.max_np_grad(mr_grad,self.val,len(self.shape)-1,m)
            #print('MAX BACKWARD',m_grad)
            self.grad = m_grad

        out._backward = __backward
        return out






class Tensor(OpsMain):
    '''
        This is the main tensor class object; 
        Call this to create tensor objects and apply respective operations desired
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

    z = y.dot(x).softmax().mean()
    print('z',z)
    z.backward()
    # print("--- %s seconds ---" % (time.time() - start_time))
    print('dz',z.grad)
    # print('db',b.grad)
    print('dy',y.grad)
    print('dx',x.grad)
    #z.history()
    
    

