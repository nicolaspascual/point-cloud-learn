from sklearn.ensemble import RandomForestClassifier
from model_selector import model_selector
import numpy as np

model_selector(
    RandomForestClassifier,
    {
        'RandomForestClassifier__n_estimators': [10, 100, 500, 1000],
        'RandomForestClassifier__criterion': ['gini', 'entropy'],
    }
)
