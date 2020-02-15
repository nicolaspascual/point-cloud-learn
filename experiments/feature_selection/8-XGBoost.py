import sys
sys.path.append('../')

from xgboost import XGBClassifier
from feature_selector import feature_selector
import numpy as np
from src.oversampling import SMOTETomekLinksDecorator,  ClusterCentroids
from configuration_extractor import extract_configuration

from imblearn.over_sampling import ADASYN

np.random.seed(0)

base_name = './model_selection/results/XGBClassifier-'
for ov in [ClusterCentroids(), SMOTETomekLinksDecorator()]:
    conf = extract_configuration(base_name + ov.__class__.__name__ + '.pickle')
    feature_selector(
        XGBClassifier(**conf),
        ov
    )
