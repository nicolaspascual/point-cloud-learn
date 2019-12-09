from sklearn.neighbors import KNeighborsClassifier
from model_selector import model_selector
import numpy as np

model_selector(
    KNeighborsClassifier,
    {
        'KNeighborsClassifier__n_neighbors': [1, 3, 5, 9]
    }
)