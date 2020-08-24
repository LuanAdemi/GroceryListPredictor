import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp # this library uses multiprocesing, since we are going to train k networks

from kmeans import kmeans

# TODO: write the class in an extra module
class KCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KCN, self).__init__()
    
        self.hidden_size = hidden_size

        self.i2h1 = nn.Linear(input_size + hidden_size, 2*(input_size + hidden_size))
        self.i2h2 = nn.Linear(2*(input_size + hidden_size), hidden_size)
        self.i2o1 = nn.Linear(input_size + hidden_size, 2*(input_size + hidden_size))
        self.i2o2 = nn.Linear(2*(input_size + hidden_size), 2*(input_size + hidden_size))
        self.i2o3 = nn.Linear(2*(input_size + hidden_size), output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h1(combined)
        hidden = self.i2h2(hidden)
        output = self.i2o1(combined)
        output = self.i2o2(output)
        output = self.i2o3(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
class knetworks:
    def __init__(self, k, data, optimize=False, verbose=False):
        super(knetworks, self).__init__()
        
        self.km = kmeans(k)
        self.km.fit(data, max_iters=1, optimize=optimize, verbose=verbose)
        self.k = self.km.k
        self.centroids = self.km.centroids
        
        self.data = data
        
        self.D = self.km.calcDistances(self.centroids, data)
        
        self.W = np.minimum((1/self.D)**2, np.full(self.D.shape, 100))
        
        self.W = [self.W[i]/sum(self.W[i]) for i in range(self.k)]
        self.W = np.array(self.W)
        
        self.networks = []
        
    def sampleRandom(self, centroid):
        return np.random.choice(np.array(range(len(self.data))),p=self.W[centroid])
    

        