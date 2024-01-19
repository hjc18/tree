import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import time

from tree.decision_tree import CARTClassifier
from tree.random_forest import RFClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def main():
    data = pd.read_csv('data/bank.csv', sep=';')
    discrete_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome',
                         'y']
    # continuous_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    le = LabelEncoder()
    for feature in discrete_features:
        data[feature] = le.fit_transform(data[feature])
    data = shuffle(data, random_state=42).reset_index(drop=True)
    X = data.drop('y', axis=1)
    y = data['y']
    models = [
            DecisionTreeClassifier(min_samples_split=264, max_depth=None),
            CARTClassifier(min_samples_split=264, max_depth=None),
            RandomForestClassifier(min_samples_split=100, max_depth=None, n_estimators=100, random_state=42),
            RFClassifier(min_samples_split=100, max_depth=None, n_estimators=100, n_jobs=6)
        ]
    kf = KFold(n_splits=5)
    for model in models:
        accuracy = []
        f1 = []
        auc = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            st = time.time()
            model.fit(X_train, y_train)
            print(f"Fold training time {time.time() - st}")
            y_pred = model.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))
            f1.append(f1_score(y_test, y_pred))
            auc.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        print(model.__class__.__name__)
        print('Mean Accuracy:', np.mean(accuracy))
        print('Mean F1:', np.mean(f1))
        print('Mean AUC:', np.mean(auc))
        print()


if __name__ == '__main__':
    main()
