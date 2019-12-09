from sklearn.tree import DecisionTreeClassifier
from model_selector import model_selector
import numpy as np

model_selector(
    DecisionTreeClassifier,
    {
        'DecisionTreeClassifier__splitter': ['best', 'random'],
        'DecisionTreeClassifier__criterion': ['gini', 'entropy'],
    }
)