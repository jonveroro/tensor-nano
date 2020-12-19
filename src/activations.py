import numpy as np

class activations:
    def relu(self,Z):
        R = np.maximum(0, Z)
        return R
    
    def sigmoid(self,Z):
        S = 1 / (1 + np.exp(-Z))
        return S
    
    def tanh(self,x):
        t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return t
    
    def d_tanh(self,t):
        dt=1-t**2
        return dt
    
    def d_sigmoid(self,Z):
        dS = Z * (1 - Z)
        return dS

    def d_relu(self,z):
        dZ= np.where(z <= 0, 0, 1)
        return dZ