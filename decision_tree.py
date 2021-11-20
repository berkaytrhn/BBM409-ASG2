from math import log

import numpy as np

from node import Node


class DecisionTree:

    def __init__(self):
        # root node
        self.head = Node()

    def chose_best_feature(self, _set, features):  # features
        # choosing best feature based on entropies and information gains

        relative_entropy, rel_num_pos, rel_num_neg = self.calculate_entropy(_set[:, -1])

        gains = []
        for column in features:
            information = 0
            for value in np.unique(_set[:, column]):
                # filter for attribute value
                temp_filter = (_set[:, column] == value)

                # using that filter to get attributes
                temp_samples = _set[temp_filter]

                # calculate entropy and number of samples
                temp_entropy, temp_pos, temp_neg = self.calculate_entropy(temp_samples[:, -1])

                # calculate ratio of relevant attribute
                temp_ratio = (temp_pos + temp_neg) / (rel_num_pos + rel_num_neg)

                # summing the result with information
                information += (temp_entropy * temp_ratio)

            gain = relative_entropy - information

            gains.append((column, gain))

        gains = sorted(gains, key=lambda x: x[1], reverse=True)
        return gains[0]

    def labelStatusCheck(self, examples):
        # checks whether all given data label is 0 or 1 or else
        pos_filter = examples[:, -1] == 1
        neg_filter = examples[:, -1] == 0

        positives = examples[pos_filter]
        negatives = examples[neg_filter]

        num_pos = len(positives)
        num_neg = len(negatives)

        if num_pos == 0:
            # all 0
            return 0
        elif num_neg == 0:
            # all 1
            return 1
        else:
            # both exists
            return -1

    def mostCommonTargetAttribute(self, examples):
        # returns most common label in given data
        pos_filter = examples[:, -1] == 1
        neg_filter = examples[:, -1] == 0

        positives = examples[pos_filter]
        negatives = examples[neg_filter]

        num_pos = len(positives)
        num_neg = len(negatives)
        return (1 if (num_pos > num_neg) else 0)

    def ID3(self, root, data, features):
        # ID3 general function, calls itself
        if not root:
            root = Node()

        if self.labelStatusCheck(data) == 1:
            # all labels are 1, becomes leaf
            root.is_leaf = True
            root.leaf_value = 1
            return root
        elif self.labelStatusCheck(data) == 0:
            # all labels are 0, becomes leaf
            root.is_leaf = True
            root.leaf_value = 0
            return root

        if len(features) == 0:
            # no feature left, leaf value becomes the most common label value
            root.is_leaf = True
            root.leaf_value = self.mostCommonTargetAttribute(data)
            return root

        # choose best feature according to its gain
        res = self.chose_best_feature(data, features)
        best_feature = res[0]
        gain = res[1]

        root.value = best_feature

        chosen_feature_values = np.unique(data[:, best_feature])
        for value in chosen_feature_values:
            # recursive calls based on children

            # configure child node
            child = Node()
            root.children.append(child)
            child.feature_value = value

            # filter data according to chosen best feature
            _filter = data[:, best_feature] == value
            new_data = data[_filter]

            # update features (remove chosen feature)
            new_features = self.update_features(features, best_feature)

            # recursive call
            self.ID3(child, new_data, new_features)

        return root

    def predict(self, X, y):
        # prediction method
        y_pred = []
        y_true = []
        for index in range(len(X)):
            X_row = X[index]
            y_row = y[index]

            # calling  traverse function for traversing and returning prediction
            pred = self.traverse(self.head, X_row)

            # predictions and labels arrays
            y_pred.append(pred)
            y_true.append(y_row[0])

        return y_pred, y_true

    def display_tree(self, node, column_map, level=0):
        # text based display function for tree
        try:
            self.display_tree((node.children)[0], column_map, level + 1)
        except IndexError:
            pass
        name = column_map[node.value] if (node.value != None) else node.leaf_value
        print(' ' * 4 * level + '->', name)
        try:
            self.display_tree((node.children)[1], column_map, level + 1)
        except IndexError:
            pass

    def update_features(self, features, chosen_best):
        # update features according to given chosen feature (remove chosen feature)
        updated = np.delete(features, np.where(features == chosen_best), axis=0)
        return updated

    def fit(self, X, y):
        # function for trigger tree build

        data = np.concatenate((X, y), axis=1)

        features = np.array(list(range(len(data[0, :-1]))))

        # first call for ID3
        self.ID3(self.head, data, features)

    def traverse(self, node, X):
        # tree traverse function

        if node.is_leaf:
            # returning prediction
            return node.leaf_value

        value = node.value
        temp_feature_value = X[value]

        for child in node.children:
            feature_value = child.feature_value
            if feature_value == temp_feature_value:
                return self.traverse(child, X)

        return (node.feature_value + 1) % 2

    def calculate_entropy(self, y):
        # calculates entropy for given set

        num_pos = len(y[y == 1])
        num_neg = len(y[y == 0])
        num_total = num_pos + num_neg

        p_ratio = num_pos / num_total
        n_ratio = num_neg / num_total

        log_p = log(p_ratio, 2) if not p_ratio == 0 else 0
        log_n = log(n_ratio, 2) if not n_ratio == 0 else 0

        entropy = (-p_ratio * log_p) - (n_ratio * log_n)

        return entropy, num_pos, num_neg
