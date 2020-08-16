from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import random

#   An implementation of the kmeans algorithm with 
#   k optimization using the elbow method.

class kmeans:
    def __init__(self, k):
        super(kmeans, self).__init__()
        
        self.k = k
        
    def initCentroids(self, k, data):
        centroids = data.copy()
        np.random.shuffle(centroids)
        return centroids[:k]
    
    def calcDistances(self, centroids, data):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        return distances
    
    def fit(self, data, max_iters=10, verbose=False, optimize=False):
        if optimize == False:
            centroids = self.initCentroids(self.k, data)

            for epoch in range(max_iters):
                distances = self.calcDistances(centroids, data)
                    
                affiliations = np.argmin(distances, axis=0)
                
                centroids = np.array([data[affiliations==k].mean(axis=0) for k in range(centroids.shape[0])])
                    
            self.centroids = centroids
            self.affiliations = affiliations
            
        else:
            best = [[-1],[],[]]
            for cK in range(self.k, self.k*3):
                centroids = self.initCentroids(cK, data)

                for epoch in range(max_iters):
                    distances = self.calcDistances(centroids, data)

                    affiliations = np.argmin(distances, axis=0)
                    #print(affiliations)

                    centroids = np.array([data[affiliations==k].mean(axis=0) for k in range(centroids.shape[0])])

                silscore = silhouette_score(data, affiliations)
                if verbose:
                    print("[%d] Elbow'in... SilScore=%f" % (cK, silscore))
                
                if best[0] < silscore:
                    best[0] = silscore
                    best[1] = cK
                    best[2] = centroids
            
            self.centroids = np.array(best[2])
            self.k = best[1]
            self.affiliations = affiliations