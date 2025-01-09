import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from sklearn.neighbors import NearestNeighbors

class Statistics:
    @staticmethod
    def compute_mmd(K_XX, K_YY, K_XY):
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        term1 = (1 / (m * (m - 1))) * (np.sum(K_XX) - np.sum(np.diagonal(K_XX)))
        term2 = (1 / (n * (n - 1))) * (np.sum(K_YY) - np.sum(np.diagonal(K_YY)))
        term3 = (2 / (m * n)) * np.sum(K_XY)

        mmd = term1 + term2 - term3

        return mmd

    @staticmethod
    def median_heuristic(X):
        pairwise_dists = pairwise_distances(X, metric='euclidean')
        median_dist = np.median(pairwise_dists)
        gamma = 1 / (2 * (median_dist ** 2))
        return gamma

    @staticmethod
    def get_mmd_score(X_train_real, synthetic_data):
        gamma = 1.0 / X_train_real.shape[1]
        K_XX = pairwise_kernels(X_train_real, X_train_real, metric='rbf', gamma=gamma)
        K_YY = pairwise_kernels(synthetic_data, synthetic_data, metric='rbf', gamma=gamma)
        K_XY = pairwise_kernels(X_train_real, synthetic_data, metric='rbf', gamma=gamma)

        return Statistics.compute_mmd(K_XX, K_YY, K_XY)
    
    @staticmethod
    def distance_to_the_closest_neighbor(X_train_real, synthetic_data, n_neighbors=1):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        nbrs.fit(X_train_real)
        
        syn_distances, syn_indices = nbrs.kneighbors(synthetic_data)
        syn_average_distance = np.round(np.mean(syn_distances), 4)
        
        return syn_average_distance
    
    @staticmethod
    def count_feature_overlap(features_synthetic, features_real):
        if isinstance(features_synthetic, dict):
            features_synthetic = set([feature for sublist in features_synthetic.values() 
                                      for feature in sublist])
    
        if isinstance(features_real, dict):
            features_real = set([feature for sublist in features_real.values() 
                                 for feature in sublist])
        overlap = features_synthetic & features_real
        overlap_proportion = len(overlap) / len(features_synthetic) if len(features_synthetic) > 0 else 0
        return len(overlap), np.round(overlap_proportion, 4)


    
