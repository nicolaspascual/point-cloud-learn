from sklearn.svm import SVC
from feature_selector import feature_selector
import numpy as np
from src.oversampling import G_SMOTEDecorator
from configuration_extractor import extract_configuration

from imblearn.over_sampling import ADASYN

np.random.seed(0)

base_name = './model_selection/results/SVC-'

for ov in [G_SMOTEDecorator(), ADASYN()]:
    conf = extract_configuration(base_name + ov.__class__.__name__ + '.pickle')
    feature_selector(
        SVC(**conf),
        ov
    )
