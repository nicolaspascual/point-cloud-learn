from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from feature_selector import feature_selector
import numpy as np
from src.oversampling import G_SMOTEDecorator
from configuration_extractor import extract_configuration

from imblearn.over_sampling import ADASYN

np.random.seed(0)

base_name = './model_selection/results/LinearDiscriminantAnalysis-'
for ov in [G_SMOTEDecorator(), ADASYN()]:
    conf = extract_configuration(base_name + ov.__class__.__name__ + '.pickle')
    feature_selector(
        LinearDiscriminantAnalysis(**conf),
        ov
    )
