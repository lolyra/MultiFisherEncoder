import torch
import numpy

class AutoEncoder:
    def __init__(self, n_components: int, n_epochs: int = 20, batch_size: int = 100, device : str = 'cpu'):
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.device = device
        self.batch_size = batch_size

    def fit(self, x, y=None):
        n_samples = x.shape[0]
        n_features = x.shape[1]
        
        self.ae = torch.nn.Sequential(
            torch.nn.Linear(n_features, self.n_components),
            torch.nn.Linear(self.n_components, n_features),
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.ae.parameters())
        loss_fn = torch.nn.MSELoss()
        for epoch in range(self.n_epochs):
            for i in range(0,n_samples,self.batch_size):
                z = torch.Tensor(x[i:i+self.batch_size]).to(self.device)
                y = self.ae(z)
                loss = loss_fn(y, z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
    def transform(self, x):
        with torch.no_grad():
            for i in range(0,x.shape[0],self.batch_size):
                o = self.ae[0](torch.Tensor(x[i:i+self.batch_size]).to(self.device)).cpu().numpy()
                if i == 0:
                    y = o
                else:
                    y = numpy.append(y,o,axis=0)
        return y


class AvgPooling:
    def __init__(self, n_components: int, batch_size: int = 500, device : str = 'cpu'):
        self.n_components = n_components
        self.device = device
        self.batch_size = batch_size

    def fit(self, x, y=None):
        n_samples = x.shape[0]
        n_features = x.shape[1]
        stride = n_features//self.n_components
        kernel = n_features-stride*self.n_components+1
        self.layer = torch.nn.AvgPool1d(kernel, stride).to(self.device)

    def transform(self, x):
        with torch.no_grad():
            for i in range(0,x.shape[0],self.batch_size):
                o = self.layer(torch.Tensor(x[i:i+self.batch_size]).to(self.device)).cpu().numpy()
                if i == 0:
                    y = o
                else:
                    y = numpy.append(y,o,axis=0)
        return y
        