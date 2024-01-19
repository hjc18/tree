import numpy as np
from concurrent.futures import ProcessPoolExecutor

from .decision_tree import CARTRegressor, CARTClassifier


class RFRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features='sqrt', random_state=42, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self.create_tree, X, y, self.random_state + n) for n in range(self.n_estimators)]
                self.trees = [future.result() for future in futures]
        else:
            for n in self.n_estimators:
                self.trees.append(self.create_tree(X, y, self.random_state + n))

    def create_tree(self, X, y, random_state):
        rng = np.random.RandomState(random_state)

        n_samples, n_features = X.shape
        # Feature bagging
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        max_features = max(1, max_features)
        # Bootstrap sample
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_sample, y_sample = X[indices], y[indices]

        features_id = rng.choice(n_features, max_features, replace=False)
        X_sample = X_sample[:, features_id]

        # Create and train a regression tree
        tree = CARTRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        tree.fit(X_sample, y_sample)
        return tree, features_id


    def predict(self, X):
        X = np.array(X)
        predictions = np.zeros((X.shape[0]))
        for tree, features_indices in self.trees:
            predictions += tree.predict(X[:, features_indices])
        predictions /= self.n_estimators
        return predictions

    def score(self, x, y_true):
        y_pred = self.predict(x)
        y_true = np.array(y_true)
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (numerator / denominator)
        return r2


class RFClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features='sqrt', random_state=42, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.n_jobs = n_jobs
        self.threshold = 0.5

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self.create_tree, X, y, self.random_state + n) for n in range(self.n_estimators)]
                self.trees = [future.result() for future in futures]
        else:
            for n in self.n_estimators:
                self.trees.append(self.create_tree(X, y, self.random_state + n))

        if self.trees[0][0].binary:
            predictions = np.zeros(X.shape[0])
            for (tree, features_id) in self.trees:
                predictions += tree.predict_proba(X[:, features_id])[:, 1]
            predictions /= self.n_estimators
            self.threshold = np.mean(y)
            y = y.astype(bool)
            best_f1 = self.compute_f1_score(y, (predictions > self.threshold))
            for thr in np.arange(0, 1, 0.02):
                f1 = self.compute_f1_score(y, (predictions > thr))
                if f1 > best_f1:
                    self.threshold = thr
                    best_f1 = f1

    def compute_f1_score(self, true_labels, predicted_labels):
        # Converting to binary values
        true_positive = np.sum((predicted_labels) & (true_labels))
        false_positive = np.sum((predicted_labels) & (~true_labels))
        false_negative = np.sum((~predicted_labels) & (true_labels))

        # Calculating precision and recall
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        # Calculating F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1_score

    def create_tree(self, X, y, random_state):
        rng = np.random.RandomState(random_state)

        n_samples, n_features = X.shape
        # Feature bagging
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        max_features = max(1, max_features)
        # Bootstrap sample
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_sample, y_sample = X[indices], y[indices]

        features_id = rng.choice(n_features, max_features, replace=False)
        X_sample = X_sample[:, features_id]

        # Create and train a regression tree
        tree = CARTClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        tree.fit(X_sample, y_sample)
        return tree, features_id

    def majority_vote(self, predictions):
        res = np.zeros(predictions.shape[0])
        for i, r in enumerate(predictions):
            values, counts = np.unique(r, return_counts=True)
            res[i] = values[np.argmax(counts)]
        return res

    def predict(self, X):
        X = np.array(X)
        if self.trees[0][0].binary:
            predictions = np.zeros(X.shape[0])
            for (tree, features_indices) in self.trees:
                predictions += tree.predict_proba(X[:, features_indices])[:, 1]
            predictions /= self.n_estimators
            return (predictions > self.threshold).astype(int)
        else:
            predictions = np.zeros((X.shape[0], self.n_estimators))
            for i, (tree, features_indices) in enumerate(self.trees):
                predictions[:, i] = tree.predict(X[:, features_indices])
            return self.majority_vote(predictions)
    
    # only for binary classification
    def predict_proba(self, X):
        assert(self.trees[0][0].binary)
        X = np.array(X)
        predictions = np.zeros((X.shape[0], 2))
        for (tree, features_indices) in self.trees:
            predictions += tree.predict_proba(X[:, features_indices])
        predictions /= self.n_estimators
        return predictions
