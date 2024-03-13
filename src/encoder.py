import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

class EncoderGMM:
    def __init__(self, n_kernels, bayesian : bool = False):
        assert n_kernels > 0
        self.n_kernels = n_kernels
        self.fitted = False
        self.sqrt2 = np.sqrt(2)
        if bayesian:
            self.model = BayesianGaussianMixture
        else:
            self.model = GaussianMixture

    def fit(self, x, y=None):
        if self.fitted:
            return self
        D = x.shape[2]
        x = x.reshape(-1,D)
        # GMM
        self.gmm = self.model(
            n_components=self.n_kernels,
            covariance_type='diag', 
            reg_covar=1e-4*x.std(axis=0).max()
        ).fit(x, y)
        self.fitted = True
        return self
    
    def transform(self, x):
        K = self.n_kernels
        N = x.shape[0]
        T = x.shape[1]
        D = x.shape[2]
        
        weights = self.gmm.weights_.astype('float32')
        means = self.gmm.means_.astype('float32').reshape(1,K,D)
        sigma = np.sqrt(self.gmm.covariances_).astype('float32').reshape(1,K,D)
        # Gamma
        p = self.gmm.predict_proba(x.reshape(-1,D)).astype('float32')*weights
        p = (p/p.sum(axis=1).reshape(-1,1)).reshape(N,T,K,1)
        # Compute Statistics
        x = x.reshape(N,T,1,D)
        s0 = p.sum(axis=1).reshape(N,K,1)
        p = p*x
        s1 = p.sum(axis=1)
        p = p*x
        s2 = p.sum(axis=1)
        del p
        
        # Compute Fisher Vector signature
        weights = weights.reshape(1,-1,1)
        v0 = (s0-T*weights)
        weights = np.sqrt(weights)
        v0 = v0/weights

        v1 = (s1-means*s0)/(weights*sigma)
        sigma*=sigma
        v2 = (s2-2*means*s1+(means**2-sigma)*s0)/(self.sqrt2*weights*sigma)

        del s0,s1,s2
        v = np.concatenate((v0,v1,v2),axis=2).reshape(N,-1)
        del v0,v1,v2

        return v