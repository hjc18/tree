import numpy as np


class Node:
    def __init__(self, best_feature, best_split_value, is_leaf=False, mean_y=None, proba=None):
        self.feature_id = best_feature
        self.split_value = best_split_value
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.mean_y = mean_y
        self.proba = proba

    def set_leaf(self, mean_y, proba=None):
        self.is_leaf = True
        self.feature_id = -1
        self.split_value = -1
        self.left_child = None
        self.right_child = None
        self.mean_y = mean_y
        self.proba = proba

    def search(self, features):
        current = self
        while not current.is_leaf:
            if features[current.feature_id] <= current.split_value:
                node = current.left_child
            else:
                node = current.right_child
            if node is None:
                return current.mean_y, current.proba
            current = node
        return current.mean_y, current.proba


class CARTRegressor():
    def __init__(self, max_depth=None, min_samples_split=2):
        self.feature_num = None
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)
        self.epsilon = 1e-15
        self.X = []
        self.Y = []

    def fit(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        if not len(Y):
            raise Exception()
        self.feature_num = X.shape[1]
        self.tree = self.build(range(X.shape[0]), 0)
        self.X, self.Y = [], []

    def build(self, sample_id_list, depth):
        nsample = len(sample_id_list)
        if nsample < self.min_samples_split \
                or (self.max_depth is not None and depth > self.max_depth):
            return Node(-1, -1, True, np.mean(self.Y[sample_id_list]))
        else:
            best_feature_id = -1
            best_split = -1
            best_left_id, best_right_id = [], []
            min_mse = float("inf")
            localY = self.Y[sample_id_list]
            mean_y = np.mean(localY)
            if np.sum((localY - mean_y)**2) < self.epsilon:
                return Node(-1, -1, True, mean_y)
            localX = self.X[sample_id_list, :]
            sample_id_list_np = np.array(sample_id_list)
            for feature_id in range(self.feature_num):
                feature = localX[:, feature_id]
                sorted_id = np.argsort(feature)
                for i in range(nsample):
                    if i < nsample - 1 and feature[sorted_id[i]] == feature[sorted_id[i + 1]]:
                        continue
                    left_id = sorted_id[:i+1]
                    right_id = sorted_id[i+1:]
                    mse = 0
                    if len(left_id):
                        mse += np.var(localY[left_id], ddof=0) * len(left_id)
                        if mse >= min_mse:
                            continue
                        left_id = sample_id_list_np[left_id]
                    if len(right_id):
                        mse += np.var(localY[right_id], ddof=0) * len(right_id)
                        right_id = sample_id_list_np[right_id]
                    if mse < min_mse:
                        min_mse = mse
                        best_feature_id = feature_id
                        best_split = feature[sorted_id[i]]
                        best_left_id = left_id
                        best_right_id = right_id
            if best_feature_id == -1 or len(best_left_id) == 0 or len(best_right_id) == 0:
                return Node(-1, -1, True, mean_y)
            node = Node(best_feature_id, best_split, False, mean_y)
            node.left_child = self.build(best_left_id, depth + 1)
            node.right_child = self.build(best_right_id, depth + 1)
            # pruning
            if node.left_child.is_leaf and node.right_child.is_leaf \
                    and abs(node.left_child.mean_y - node.right_child.mean_y) < self.epsilon:
                node.set_leaf(node.left_child.mean_y)
            return node

    def predict(self, X):
        return np.array([self.tree.search(x)[0] for x in np.array(X)])

    def score(self, x, y_true):
        y_pred = self.predict(x)
        y_true = np.array(y_true)
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (numerator / denominator)
        return r2


class CARTClassifier():
    def __init__(self, max_depth=None, min_samples_split=2):
        self.feature_num = None
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)
        self.epsilon = 1e-15
        self.binary = False
        self.X = []
        self.Y = []

    def fit(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        if not len(Y):
            raise Exception()
        self.feature_num = self.X.shape[1]
        self.classes = np.unique(self.Y)
        self.binary = len(self.classes) == 2
        self.tree = self.build(range(X.shape[0]), 0)
        self.X, self.Y = [], []

    def major_class(self, Y):
        if self.binary:
            cnt = np.count_nonzero(Y == self.classes[0])
            return (self.classes[0], cnt) if cnt * 2 >= len(Y) else (self.classes[1], len(Y) - cnt)
        values, counts = np.unique(Y, return_counts=True)
        a = np.argmax(counts)
        return values[a], counts[a]

    def gini(self, Y):
        if self.binary:
            p = np.count_nonzero(Y == self.classes[0]) / len(Y)
            return 2 * p * (1 - p)
        _, counts = np.unique(Y, return_counts=True)
        return 1 - np.sum(counts ** 2) / len(Y) ** 2

    def build(self, sample_id_list, depth):
        nsample = len(sample_id_list)
        if nsample < self.min_samples_split \
                or (self.max_depth is not None and depth > self.max_depth):
            mean_y, cnt = self.major_class(self.Y[sample_id_list])
            return Node(-1, -1, True, mean_y, proba=cnt/nsample if mean_y else 1-cnt/nsample)
        else:
            localY = self.Y[sample_id_list]
            mean_y, cnt = self.major_class(localY)
            proba = cnt/nsample if mean_y else 1-cnt/nsample
            if cnt == nsample:
                # all samples share the same Y
                return Node(-1, -1, True, mean_y, proba=proba)
            best_feature_id = -1
            best_split = -1
            best_left_id, best_right_id = [], []
            min_mse = float("inf")
            localX = self.X[sample_id_list, :]
            sample_id_list_np = np.array(sample_id_list)
            for feature_id in range(self.feature_num):
                feature = localX[:, feature_id]
                sorted_id = np.argsort(feature)
                for i in range(nsample):
                    if i < nsample - 1 and feature[sorted_id[i]] == feature[sorted_id[i + 1]]:
                        continue
                    left_id = sorted_id[:i + 1]
                    right_id = sorted_id[i + 1:]
                    mse = 0
                    if len(left_id):
                        mse += self.gini(localY[left_id]) * len(left_id) / nsample
                        if mse >= min_mse:
                            continue
                        left_id = sample_id_list_np[left_id]
                    if len(right_id):
                        mse += self.gini(localY[right_id]) * len(right_id) / nsample
                        right_id = sample_id_list_np[right_id]
                    if mse < min_mse:
                        min_mse = mse
                        best_feature_id = feature_id
                        best_split = feature[sorted_id[i]]
                        best_left_id = left_id
                        best_right_id = right_id
            if best_feature_id == -1 or len(best_left_id) == 0 or len(best_right_id) == 0:
                return Node(-1, -1, True, mean_y, proba=proba)
            node = Node(best_feature_id, best_split, False, mean_y, proba=proba)
            node.left_child = self.build(best_left_id, depth + 1)
            node.right_child = self.build(best_right_id, depth + 1)
            return node

    def predict(self, X):
        res = np.array([self.tree.search(x)[0] for x in np.array(X)])
        return res
    
    # only for binary classification
    def predict_proba(self, X):
        assert(self.binary)
        res = np.array([self.tree.search(x)[1] for x in np.array(X)])
        return np.column_stack((1-res, res))
