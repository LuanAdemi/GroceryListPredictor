import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp # this library uses multiprocesing, since we are going to train k networks

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
    
def knetworks():
    def __init__(self, k, centroids, affiliations, vocabulary):
        super(knetworks, self).__init__()
        
        self.k = k
        self.centroids = centroids
        self.vocabulary = vocabulary
        self.vocabLen = len(vocabulary)
        
        self.networks = []
        
        # create k network which later will be specialized on the data that is in the cluster
        for i in range(self.k):
            model = KCN(self.vocabLen, 512, self.vocabLen)
            self.networks.append(model)
        
    def distance(self, p1, p2):
        return np.sum((p1 - p2)**2) 
    
    def getNearestCentroid(self, point):
        distances = []
        for i in range(self.k):
            distances.append(distance(point, centroid[i]))
        
        distances = np.array(distances)
        return distances.argmin()
    
    def fit(self, data):
        for input, output in data:
            
    def train(self, data):
        mp = torch.multiprocessing.get_context('forkserver')
        