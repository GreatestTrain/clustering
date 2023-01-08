import numpy as np
from scipy.spatial import distance
from copy import deepcopy

class KMeans:

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.train_data_ = None
        self.distances_ = None

    @property
    def n_features(self):
        return np.shape(self.train_data)[1]

    @property
    def train_data(self):
        return self.train_data_
        # self.number_features = 
    @train_data.setter
    def train_data(self, x: np.ndarray):
        self.train_data_ = x

    def initialize_centers(self):
        self.centers = np.random.uniform(low=np.min(self.train_data, axis=0), high=np.max(self.train_data, axis=0) ,size=(self.n_clusters, self.n_features))
        return self.centers
    
    @property
    def distances(self):
        return self.get_distances()
    @property
    def min_distances(self):
        return self.get_min_distances()
    @property
    def labels(self):
        return self.get_labels()

    def get_distances(self):
        self.distances_ = distance.cdist(self.train_data, self.centers)
        return self.distances_
    
    def get_min_distances(self):
        self.min_distances_ = np.min(self.distances, axis=1)
        return self.min_distances_

    def get_labels(self):
        self.labels_ = np.argmin(self.distances, axis=1)
        return self.labels_
    
    def cost(self):
        self.cost_ = np.sum(self.min_distances ** 2)
        return self.cost_
    
    def gradient(self, delta: float):
        og_shape = np.shape(self.centers)
        old_centers = self.centers.ravel()
        gradient = np.empty_like(old_centers)

        _p = np.copy(old_centers)
        for idx, dim in enumerate(old_centers):
            _p[idx] += delta
            new = deepcopy(self)
            new.centers = np.reshape(_p, newshape=og_shape)
            gradient[idx] = (new.cost() - self.cost()) / delta
            _p = np.copy(old_centers)

        return np.reshape(gradient, og_shape)

    def optimize(self, n_iterations: int = 1000, lr: float =0.01):
        
        for _ in range(n_iterations):
            diff = - lr * self.gradient(0.01)
            if np.all(np.abs(diff) <= 1e-6):
                break

            self.centers += diff

            self.get_distances()
            self.get_min_distances()