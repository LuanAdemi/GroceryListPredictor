import numpy as np

from kmeans import kmeans

class knetworks:
    def __init__(self, k, data, optimize=False, verbose=False):
        super(knetworks, self).__init__()
        
        self.km = kmeans(k)
        self.km.fit(data, max_iters=1, optimize=optimize, verbose=verbose)
        self.k = self.km.k
        self.centroids = self.km.centroids
        
        self.data = data
        
        self.D = self.km.calcDistances(self.centroids, data)
        
        self.W = np.minimum((1/self.D)**2, np.full(self.D.shape, 50))
        
        self.W = [self.W[i]/sum(self.W[i]) for i in range(self.k)]
        self.W = np.array(self.W)
        
        self.networks = []
        
    def sampleRandom(self, centroid):
        return np.random.choice(np.array(range(len(self.data))),p=self.W[centroid])
    

        