from math import log
import numpy as np
from numpy.lib.shape_base import column_stack
import pandas as pd
from node import Node


class DecisionTree:



    def __init__(self):
        self.head = Node()


    def chose_best_feature(self, _set, features): #features
        # fetaure based entropies and information gains

        relative_entropy, rel_num_pos, rel_num_neg = self.calculate_entropy(_set[:,-1])

        gains = []
        for column in features:
        #print("new features: ",features)
        #for feature in features:
            information = 0
            for value in np.unique(_set[:,column]):
                # filter for attribute value
                temp_filter = (_set[:,column] == value)
                
                # using that filter to get attributes
                temp_samples = _set[temp_filter]

                # calculate entropy and number of samples
                temp_entropy, temp_pos, temp_neg = self.calculate_entropy(temp_samples[:,-1])
                
                # calculate ratio of relevant attribute
                temp_ratio = (temp_pos+temp_neg)/(rel_num_pos+rel_num_neg)
                
                # summing the result with information
                information += (temp_entropy*temp_ratio)
                #print((temp_entropy*temp_ratio))
            
            gain = relative_entropy-information

            #print(f"For column {column}, gain: {gain}")
            gains.append((column, gain))
        
        gains = sorted(gains, key=lambda x:x[1], reverse=True)
        #print(gains)
        return gains[0]

    def chose_best_feature_old(self, _set): #features
        # fetaure based entropies and information gains

        relative_entropy, rel_num_pos, rel_num_neg = self.calculate_entropy(_set[:,-1])

        gains = []
        for column in range(_set.shape[1]-1):
        #print("new features: ",features)
        #for feature in features:
            information = 0
            for value in np.unique(_set[:,column]):
                # filter for attribute value
                temp_filter = (_set[:,column] == value)
                
                # using that filter to get attributes
                temp_samples = _set[temp_filter]

                # calculate entropy and number of samples
                temp_entropy, temp_pos, temp_neg = self.calculate_entropy(temp_samples[:,-1])
                
                # calculate ratio of relevant attribute
                temp_ratio = (temp_pos+temp_neg)/(rel_num_pos+rel_num_neg)
                
                # summing the result with information
                information += (temp_entropy*temp_ratio)
                #print((temp_entropy*temp_ratio))
            
            gain = relative_entropy-information

            #print(f"For column {column}, gain: {gain}")
            gains.append((column, gain))
        
        gains = sorted(gains, key=lambda x:x[1], reverse=True)
        #print(gains)
        return gains[0]

    def labelStatusCheck(self, examples):
        pos_filter = examples[:,-1] == 1
        neg_filter = examples[:,-1] == 0
        
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
        """
        posCount = 0
        negCount = 0

        for i in examples[:, -1].tolist():
            if i == 1:
                posCount = posCount + 1
            else:
                negCount = negCount + 1
        """
        pos_filter = examples[:,-1] == 1
        neg_filter = examples[:,-1] == 0
        
        positives = examples[pos_filter]
        negatives = examples[neg_filter]

        num_pos = len(positives)
        num_neg = len(negatives)
        return (1 if (num_pos > num_neg) else 0)


    def ID3(self, root, data, features):

        print("*****************************")
        if not root: # may be redundant
            print("node created from if not root")
            root = Node()
        
        if self.labelStatusCheck(data) == 1:
            root.is_leaf = True
            root.leaf_value = 1
            print(f"all 1 leaf -> {root}")
            return root
        elif self.labelStatusCheck(data) == 0:
            root.is_leaf = True
            root.leaf_value = 0
            print(f"all 0 leaf -> {root}")
            return root

        if len(features) == 0:
            root.is_leaf = True
            root.leaf_value = self.mostCommonTargetAttribute(data)
            print(f"most common label leaf -> {root}")
            return root
        
        #best_feature = self.chose_best_feature(data, features)[0]
        res = self.chose_best_feature(data, features)
        best_feature = res[0]
        gain = res[1]

        root.value = best_feature

        chosen_feature_values = np.unique(data[:,best_feature])
        for value in chosen_feature_values:
            child = Node()
            root.children.append(child)
            child.feature_value = value
            child.gain = gain
            print(f"Not leaf, node -> {root}, child -> {child}")
            _filter = data[:,best_feature] == value
            
            new_data = data[_filter]
            #new_data = np.concatenate((new_data[:,:best_feature], new_data[:,best_feature+1:]), axis=1)

            #new_features = self.get_columns(new_data)
            new_features = self.update_features(features, best_feature)
            print(f"features: {features}, chosen: {best_feature}")
            #new_features = np.delete(features, np.where(features==best_feature), axis=0)
            self.ID3(child, new_data, new_features)
        
        return root



    def display_tree(self, node, column_map, level=0):
        #print(column_map)

        if len(node.children) == 2:
            child_1, child_2 = node.children
            if child_1.is_leaf and child_2.is_leaf:
                print(f"Twig, children:  {child_1.leaf_value}, {child_2.leaf_value}, gain: {node.gain}")
        elif len(node.children) == 1:
            child_1 = node.children[0]
            if child_1.is_leaf:
                print(f"Twig, children:  {child_1.leaf_value}, gain: {node.gain}")
 
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
        

    def get_columns(self, data):
        return np.array(list(range(len(data[0,:-1]))))

    def update_features(self, features, chosen_best):
        updated = np.delete(features, np.where(features == chosen_best), axis=0)
        return updated

    def fit(self, X, y):
        # setting attributes
        # features
        self.X = X
        # labels
        self.y = y

        data = np.concatenate((X,y), axis=1)


        attributes = self.get_columns(data)


        self.ID3(self.head, data, attributes)


    def traverse(self, node, X, y):
        print(self.head)

        if node.is_leaf:
            print(f"predicted: {node.leaf_value}, target: {y}")
            return node.leaf_value == y

        print(f"parent value: {node.value}")
        print(f"Current X: {X}")
        value = node.value
        temp_feature_value = X[value]

        for child in node.children:
            feature_value = child.feature_value
            print(f"child_value: {child.feature_value}")
            if feature_value == temp_feature_value:
                #X = np.delete(X, temp_feature_value, axis=0)
                return self.traverse(child, X, y)

    def predict(self, X, y):
        return self.traverse(self.head, X, y)

    def calculate_entropy(self, y):

        num_pos = len(y[y == 1])
        num_neg = len(y[y == 0])
        num_total = num_pos+num_neg
        
        p_ratio = num_pos/num_total
        n_ratio = num_neg/num_total

        log_p = log(p_ratio, 2) if not p_ratio == 0 else 0
        log_n = log(n_ratio, 2) if not n_ratio == 0 else 0

        entropy = (-p_ratio*log_p)-(n_ratio*log_n)

        return entropy, num_pos, num_neg
