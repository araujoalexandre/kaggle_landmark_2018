""" overload of kmeans class Faiss library """

import sys
import pickle
from os.path import join

sys.path.append('/home/alexandrearaujo/libraries/faiss/')
import numpy as np
import faiss


class Kmeans:
    
    def __init__(self, d, k, niter=25, verbose=False, 
                 max_points_per_centroid=None, gpu=False):
        self.d = d
        self.k = k
        self.cp = faiss.ClusteringParameters()
        self.cp.niter = niter
        self.cp.verbose = verbose
        self.centroids = None
        self.max_points_per_centroid = max_points_per_centroid
        self.gpu = gpu
        self.is_trained = False
        
    def _train_without_gpu(self, x):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d
        
        clus = faiss.Clustering(d, self.k, self.cp)
        # otherwise the kmeans implementation sub-samples the training set
        if self.max_points_per_centroid is not None:
            clus.max_points_per_centroid = self.max_points_per_centroid

        self.index = faiss.IndexFlatL2(d)
            
        clus.train(x, self.index)
        centroids = faiss.vector_float_to_array(clus.centroids)
        self.centroids = centroids.reshape(self.k, d)
        self.obj = faiss.vector_float_to_array(clus.obj)
        self.is_trained = True
        return self.obj[-1]
        
    def _train_with_gpu(self, x):
        print('training kmeans with gpu')

        d = self.d
        clus = faiss.Clustering(d, self.k, self.cp)

        # otherwise the kmeans implementation sub-samples the training set
        if self.max_points_per_centroid is not None:
            clus.max_points_per_centroid = self.max_points_per_centroid

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        self.index = faiss.GpuIndexFlatL2(res, d, cfg)

        # perform the training
        clus.train(x, self.index)
        self.centroids = faiss.vector_float_to_array(clus.centroids)

        self.obj = faiss.vector_float_to_array(clus.obj)
        self.is_trained = True
        return self.obj[-1]

    def _assign_with_gpu(self, x):
        assert self.centroids is not None, "should train before assigning"
        index = faiss.IndexFlatL2(self.d)
        self.centroids = self.centroids.reshape(self.k, self.d)
        index.add(self.centroids)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        D, I = gpu_index.search(x, 1)
        return D.ravel(), I.ravel()

    def _assign_without_gpu(self, x):
        assert self.centroids is not None, "should train before assigning"
        index = faiss.IndexFlatL2(self.d)
        self.centroids = self.centroids.reshape(self.k, self.d)
        index.add(self.centroids)
        D, I = index.search(x, 1)
        return D.ravel(), I.ravel()

    def train(self, x):
        if self.gpu:
            return self._train_with_gpu(x)
        return self._train_without_gpu(x)
    
    def assign(self, x):
        if self.gpu:
            return self._assign_with_gpu(x)
        return self._assign_without_gpu(x)

    def save(self, path):
        """function to dump picke object"""
        with open(join(path, 'kmeans.dump'), 'wb') as f:
            pickle.dump(self.centroids, f, -1)

    def load(self, path):
        """function to load pickle object"""
        with open(path, 'rb') as f:
            self.centroids = pickle.load(f, encoding='latin1')
        self.is_trained = True

