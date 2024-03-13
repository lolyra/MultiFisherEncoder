import os
import sys
import numpy as np
import pickle as pk
from copy import deepcopy

from src.encoder import EncoderGMM

class MultiFisherEncoder:
    def __init__(self, 
                 extractor: callable,
                 reducer: object,
                 reducer_args : dict = {},
                 n_kernels : int = 16,
                 bayesian_mixture : bool = False,
                 include_fc : bool = False,
                 batch_size: int = 20,
                 path : str = ''):

        self.extractor = extractor
        self.rclass = reducer
        self.rargs = reducer_args
        self.encoder = EncoderGMM(n_kernels, bayesian_mixture)
        self.include_fc = include_fc
        self.batch_size = batch_size
        self.ncomponents = sys.maxsize
        self.path = path
        self.reducer = {}
    

    def fit(self, x):
        x = self.extract_features(x)
        self.learn_dictionary(x)
        print("Done")

    def transform(self, x):
        y = None
        for i in range(0,len(x),self.batch_size):
            z = self.batch_transform(x[i:i+self.batch_size])
            if y is None:
                y = z
            else:
                y = np.concatenate((y,z),axis=0)
        return y
    
    def batch_transform(self, x):
        y = self.extractor(x)
        n = len(y) - (1 if self.include_fc else 0)
        for j in range(n):
            z = y[j].reshape(y[j].shape[0],y[j].shape[1],-1).swapaxes(1,2)
            z = self.reduce_dimension(z,j)
            if j == 0:
                x = z
            else:
                x = np.concatenate((x,z),axis=1)
        x = self.encoder.transform(x)
        if self.include_fc:
            x = np.concatenate((x,y[-1]),axis=1)
        # Normalization (Improved Fisher Vector)
        x = np.sign(x)*np.sqrt(np.fabs(x))
        x = x / np.linalg.norm(x, axis=1).reshape(-1,1)
        return x

    def reduce_dimension(self, x, layer:int):
        s = x.shape
        if s[2] == self.ncomponents:
            return x
        x = x.reshape(-1,s[2])
        if layer not in self.reducer:
            filename = os.path.join(self.path,f'dim_{layer}.pkl')
            if os.path.exists(filename):
                self.reducer[layer] = pk.load(open(filename,'rb'))
            else:
                self.reducer[layer] = self.rclass(
                    n_components = self.ncomponents, 
                    **self.rargs
                )
                self.reducer[layer].fit(x)
                pk.dump(self.reducer[layer], open(filename,"wb"))
        x = self.reducer[layer].transform(x).reshape(s[0],-1,self.ncomponents)
        return x

    def extract_features(self, x):
        print("Extracting local features from convolutional layers")
        y = {}
        total = len(x)
        for i in range(0,total,self.batch_size):
            inputs = x[i:i+self.batch_size]
            output = self.extractor(inputs)
            n = len(output) - (1 if self.include_fc else 0)
            for j in range(n):
                aux = output[j]
                aux = aux.reshape(aux.shape[0],aux.shape[1],-1).swapaxes(1,2)
                if j not in y:
                    y[j] = np.zeros((total,*aux.shape[1:]))
                y[j][i:i+self.batch_size] = aux
        # Get Number of components
        for layer in y:
            if self.ncomponents > y[layer].shape[2]:
                self.ncomponents = y[layer].shape[2]
        # Learn dimensionality reduction
        print("Learning dimension reduction")
        for layer in y:
            z = self.reduce_dimension(y[layer],layer)
            if layer == 0:
                x = z
            else:
                x = np.concatenate((x,z),axis=1)
        return x


    def learn_dictionary(self, x):
        gmmfile = os.path.join(self.path,f"gmm.pkl")
        print("Estimating GMM")
        if not os.path.exists(gmmfile):
            self.encoder.fit(x)
            pk.dump(self.encoder.gmm, open(gmmfile,"wb"))
        else:
            self.encoder.gmm = pk.load(open(gmmfile,'rb'))
            self.encoder.fitted = True
