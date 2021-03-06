from sklearn.naive_bayes import GaussianNB
from feature_selector import feature_selector
import numpy as np
from src.oversampling import G_SMOTEDecorator
from configuration_extractor import extract_configuration

from imblearn.over_sampling import ADASYN

np.random.seed(0)

base_name = './model_selection/results/GaussianNB-'
for ov in [G_SMOTEDecorator(), ADASYN()]:
    conf = extract_configuration(base_name + ov.__class__.__name__ + '.pickle')
    feature_selector(
        GaussianNB(**conf),
        ov
    )
