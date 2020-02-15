from imblearn.base import BaseSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import _random_state_docstring

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class ClusterCentroids(BaseSampler):
    """Perform under-sampling using SV sampling
    approach for imbalanced datasets.
    """
    _sampling_type = 'under-sampling'

    def _fit_resample(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        majority_class = values[np.argmax(counts)]
        majority_class_selector = (y == majority_class)


        num_clusters = min(counts)

        clusters = KMeans(n_clusters=num_clusters)\
                        .fit(X[majority_class_selector])\
                        .cluster_centers_

        y = y[~majority_class_selector]
        X = X[~majority_class_selector]

        X = np.concatenate([X, clusters])
        y = np.concatenate([y, [majority_class for _ in range(num_clusters)]])
        
        return X, y