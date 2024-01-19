import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

import time

from torch import rand

from tree.decision_tree import CARTRegressor
from tree.random_forest import RFRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def main():
    pd_house = pd.read_csv('data/HousingData.csv')
    pd_house_median = pd_house.fillna(pd_house.median())
    pd_house_median = shuffle(pd_house_median, random_state=42).reset_index(drop=True)

    X = pd_house_median.drop('MEDV', axis=1)
    y = pd_house_median['MEDV']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    models = [
            DecisionTreeRegressor(max_depth=None, min_samples_split=45),
            CARTRegressor(max_depth=None, min_samples_split=45),
            RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=3, random_state=42),
            RFRegressor(n_estimators=100, max_depth=None, min_samples_split=3, random_state=42, n_jobs=6)
        ]

    kf = KFold(n_splits=5)
    for model in models:
        test_r2, test_mean_absolute_score, test_mean_suqared_score = [], [], []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            st = time.time()
            model.fit(X_train, y_train)
            print(f"Fold training time {time.time() - st}")
            y_pred = model.predict(X_test)
            test_r2.append(model.score(X_test, y_test))
            test_mean_absolute_score.append(mean_absolute_error(y_test, y_pred))
            test_mean_suqared_score.append(mean_squared_error(y_test, y_pred))
        print(model.__class__.__name__)
        print('Mean R2: ', np.mean(test_r2))
        print('Mean Absolute Error: ', np.mean(test_mean_absolute_score))
        print('Mean Squared Error: ', np.mean(test_mean_suqared_score))
        print()


if __name__ == "__main__":
    main()
