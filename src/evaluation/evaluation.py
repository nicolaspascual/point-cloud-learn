from sklearn.metrics import matthews_corrcoef
import pandas as pd

def imbalanced_score(recall_rate):
    def _imbalanced_score(estimator, X, y):
        
        y_pred = estimator.predict(X)
        y_true = y
        
        TP = sum(y_pred & y_true)
        FN = sum(~y_pred & y_true)
        FP = sum(y_pred & ~y_true)
        
        precision = TP / (TP + FP) 
        recall = TP / (TP + FN)

        return recall * recall_rate + precision * (1 - recall_rate)
    return _imbalanced_score

def matthews_corrcoef_score(estimator, X, y):
    y_pred = estimator.predict(X)
    y_true = y

    return matthews_corrcoef(y_true, y_pred)

nemenyi_critical_values = pd.DataFrame(
  [
    [3.884, 3.914, 3.941, 3.967, 3.992, 4.015, 4.037, 4.057, 4.077, 4.096, 4.114, 4.132, 4.148, 4.164],
    [3.426, 3.458, 3.489, 3.517, 3.544, 3.569, 3.593, 3.616, 3.637, 3.658, 3.678, 3.696, 3.714, 3.732],
    [3.196, 3.230, 3.261, 3.291, 3.319, 3.346, 3.371, 3.394, 3.417, 3.439, 3.459, 3.479, 3.498, 3.516]
  ],
  columns=[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
  index=[0.01, 0.05, 0.10],
)