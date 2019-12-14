from xgboost import XGBClassifier
from model_selector import model_selector
import numpy as np

model_selector(
    XGBClassifier,
    {
        'XGBClassifier__learning_rate': [10**-i for i in range(1, 4)],
        'XGBClassifier__n_estimators': [10, 50, 100],
        'XGBClassifier__booster': ['gbtree', 'gblinear', 'dart']
    }
)
