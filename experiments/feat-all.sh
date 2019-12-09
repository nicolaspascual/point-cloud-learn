#!/bin/zsh
source ~/.zshrc


py ./feature_selection/1-SVC.py
py ./feature_selection/2-RandomForest.py
py ./feature_selection/3-GaussianNB.py
py ./feature_selection/4-DecisionTreeClassifier.py
py ./feature_selection/5-LDA.py
py ./feature_selection/6-KNN.py
py ./feature_selection/7-AdaBoostClassifier.py
py ./feature_selection/8-XGBoost.py