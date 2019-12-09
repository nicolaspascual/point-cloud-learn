from sklearn.naive_bayes import GaussianNB
from model_selector import model_selector
import numpy as np

model_selector(
    GaussianNB,
    {}
)