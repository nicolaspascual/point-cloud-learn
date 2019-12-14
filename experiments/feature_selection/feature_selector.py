from imblearn.over_sampling import ADASYN
from src.oversampling import G_SMOTEDecorator
from src.evaluation import imbalanced_score, matthews_corrcoef_score
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
import pandas as pd
import pickle
import sys
sys.path.append('../')


def feature_selector(model, oversampler, X_path='../data/all_X.csv', y_path='../data/all_y.csv', recall_rate=0.7):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path, header=None).T.ix[0]

    options = {
        'FEATURE_SELECTION__k': list(range(5, len(X.columns)))
    }

    i_train, i_test = next(StratifiedShuffleSplit(
        test_size=0.2, random_state=40).split(X, y))

    X_train, X_test, y_train, y_test = X.loc[i_train], X.loc[i_test], y.loc[i_train], y.loc[i_test]

    model_name = model.__class__.__name__.split('.')[-1]
    pipeline = Pipeline([
        ('OVERSAMPLER', oversampler),
        ('FEATURE_SELECTION', SelectKBest(mutual_info_classif)),
        (model_name, model)
    ])

    clf = GridSearchCV(
        pipeline, options, scoring=imbalanced_score(recall_rate=recall_rate),
        n_jobs=3, verbose=10, return_train_score=True
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    with open(f'feat-results/{model_name}-{oversampler.__class__.__name__}.pickle', 'wb') as f:
        pickle.dump(clf.best_params_, f)
    with open(f'feat-results/{model_name}-{oversampler.__class__.__name__}-results.pickle', 'wb') as f:
        pickle.dump((clf.cv_results_, y_test, y_pred), f)

    print(f'{model_name}-{oversampler.__class__.__name__}')

    print(classification_report(y_test, y_pred))

    print(len(y_pred), sum(y_test), sum(y_pred), sum(y_pred & y_test))
