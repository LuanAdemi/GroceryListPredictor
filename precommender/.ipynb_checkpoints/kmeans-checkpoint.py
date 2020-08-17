from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import random
import sys

#   An implementation of the kmeans++ (https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) algorithm with 
#   k optimization using the elbow method.

class kmeans:
    def __init__(self, k):
        super(kmeans, self).__init__()
        
        self.k = k
        
    def distance(self, p1, p2): 
        return np.sum((p1 - p2)**2) 
        
    def initCentroids(self, k, data):
        centroids = []
        centroids.append(data[np.random.randint( 
            data.shape[0]), :])
        if self.verbose:
            print("Searching for best starting centroids...")
            print("[1] Setting centroid to", centroids[0])
        for c_id in range(k-1):
            dist = [] 
            for point in data: 
                d = sys.maxsize 
                for j in range(len(centroids)): 
                    temp_dist = self.distance(point, centroids[j]) 
                    d = min(d, temp_dist) 
                dist.append(d) 
                
            dist = np.array(dist) 
            next_centroid = data[np.argmax(dist), :] 
            centroids.append(next_centroid)
            if self.verbose:
                print("[%d] Setting centroid to" % (c_id+2), centroids[c_id+1])
            dist = [] 
        centroids = np.array(centroids)
        return centroids
    
    def calcDistances(self, centroids, data):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        return distances
    
    def fit(self, data, max_iters=10, verbose=False, optimize=False):
        self.verbose = verbose
        if optimize == False:
            if self.verbose:
                print("Finding the best starting centroids...")
            centroids = self.initCentroids(self.k, data)

            for epoch in range(max_iters):
                if self.verbose:
                    print("[%d] KMean'in..." % epoch)
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
                if self.verbose:
                    print("[%d] Elbow'in... SilScore=%f" % (cK, silscore))
                
                if best[0] < silscore:
                    best[0] = silscore
                    best[1] = cK
                    best[2] = centroids
            
            self.centroids = np.array(best[2])
            self.k = best[1]
            self.affiliations = affiliations