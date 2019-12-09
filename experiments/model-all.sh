#!/bin/zsh
source ~/.zshrc


py ./model_selection/1-SVC.py
py ./model_selection/2-RandomForest.py
py ./model_selection/3-GaussianNB.py
py ./model_selection/4-DecisionTreeClassifier.py
py ./model_selection/5-LDA.py
py ./model_selection/6-KNN.py
py ./model_selection/7-AdaBoostClassifier.py
py ./model_selection/8-XGBoost.py