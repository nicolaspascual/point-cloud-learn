from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from model_selector import model_selector
import numpy as np

model_selector(
    AdaBoostClassifier,
    {
        'AdaBoostClassifier__n_estimators': [10, 50, 100, 500],
        'AdaBoostClassifier__base_estimator': [DecisionTreeClassifier(max_depth=5)]
    }
)
