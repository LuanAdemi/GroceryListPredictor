from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import random

#   An implementation of the kmeans algorithm with 
#   k optimization using the elbow method.

# BUG: for k>7, the implementation breaks -> maybe use data structure that is better at handeling nans

class kmeans:
    def __init__(self, k):
        super(kmeans, self).__init__()
        
        self.k = k
        
    def initCentroids(self, k, data):
        centroids = np.zeros((k,2))
        for i in range(k):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            centroids[i] = [x,y]
        return centroids
    
    def calcDistance(self, centroid, X, Y):
        distances = []
    
        c_x = centroid[0]
        c_y = centroid[1]
    
        for x,y in list(zip(X,Y)):
            root_diff_x = (x - c_x) ** 2
            root_diff_y = (y - c_y) ** 2
            distance = np.sqrt(root_diff_x + root_diff_y)
            distances.append(distance)
        
        return distances
    
    def fit(self, data, max_iters=10, verbose=False, optimize = False):
        if optimize == False:
            distances = np.zeros((self.k,len(data)))
            affiliations = np.zeros((len(data),1))
            centroids = self.initCentroids(self.k, data)

            for epoch in range(max_iters):
                for i in range(self.k):
                    distance = self.calcDistance(centroids[i], data[:,0], data[:,1])
                    distances[i] = distance
                    
                for p in range(len(data)):
                    affiliations[p] = distances[:, p].argmin()

                affiliations = np.array(affiliations)

                for i in range(self.k):
                    x_new = np.expand_dims(data[:,0], axis=1)[affiliations == i].mean()
                    y_new = np.expand_dims(data[:,1], axis=1)[affiliations == i].mean()
                    centroids[i] = [x_new,y_new]
                    
            self.centroids = centroids
            self.affiliations = affiliations
            
        else:
            best = [[-1],[],[]]
            
            for cK in range(self.k, self.k*2):
                print(cK)
                distances = np.zeros((cK,len(data)))
                affiliations = np.zeros((len(data),1))
                centroids = self.initCentroids(cK, data)

                for epoch in range(max_iters):
                    for i in range(cK):
                        distances[i] = self.calcDistance(centroids[i], data[:,0], data[:,1])

                    for p in range(len(data)):
                        affiliations[p] = distances[:, p].argmin()

                    affiliations = np.array(affiliations)

                    for i in range(cK):
                        x_new = data[np.squeeze(affiliations, axis=1) == i][:,0].mean()
                        y_new = data[np.squeeze(affiliations, axis=1) == i][:,1].mean()
                        centroids[i] = [x_new,y_new]

                silscore = silhouette_score(data, np.squeeze(affiliations, axis=1).astype(int))
                
                if best[0] < silscore:
                    best[0] = silscore
                    best[1] = cK
                    best[2] = centroids
                    print(silscore)
            
            self.centroids = np.array(best[2])
            self.k = best[1]