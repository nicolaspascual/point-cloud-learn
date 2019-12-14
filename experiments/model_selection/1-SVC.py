from sklearn.svm import SVC
from model_selector import model_selector
import numpy as np

np.random.seed(0)

model_selector(
    SVC,
    {
        'SVC__kernel': ['poly', 'rbf', 'sigmoid'],
        'SVC__C':  [0.1, 1, 10, 100],
        'SVC__degree': np.arange(2, 8)
    }
)
