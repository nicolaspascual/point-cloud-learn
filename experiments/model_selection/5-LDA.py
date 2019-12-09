from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from model_selector import model_selector
import numpy as np

model_selector(
    LinearDiscriminantAnalysis,
    {
        'LinearDiscriminantAnalysis__solver': ['svd', 'lsqr', 'eigen'],
        'LinearDiscriminantAnalysis__shrinkage': ['none', 'auto']
    }
)