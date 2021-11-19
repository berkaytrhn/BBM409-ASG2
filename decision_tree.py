from math import log
import numpy as np
from numpy.lib.shape_base import column_stack
import pandas as pd
from node import Node


class DecisionTree:



    def __init__(self):
        self.head = Node()

    def chose_best_feature(self, _set):
        # fetaure based entropies and information gains

        relative_entropy, rel_num_pos, rel_num_neg = self.calculate_entropy(_set)

        gains = []
        for column in range(_set.shape[1]-1):
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
        # print(len(gains))
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
        if not root:
            root = Node()

        pass

    def get_columns(self, data):
        return np.array(list(range(len(data[0,:-1]))))

    def fit(self, X, y):
        # setting attributes
        # features
        self.X = X
        # labels
        self.y = y

        data = np.concatenate((X,y), axis=1)


        attributes = self.get_columns(data)


        self.ID3(self.head, data, attributes)
        """
        # all dataset
        self.data = data
        # number of positive examples
        self.num_pos = num_pos
        # number of negative examples
        self.num_neg = num_neg
        # all dataset's entropy
        self.dataset_entropy = dataset_entropy
        
        chosen_feature = self.calculate_gains(data) # (chosen, gain)
        print(chosen_feature)
        """


    def predict(self, X, y):
        temp = self.head 
        while not temp.is_leaf:
            print(temp)
            attribute = temp.value
            print(attribute)
            value = X[attribute]
            print(value)
            _next = temp.children[value]
            temp = _next
            X = np.delete(X, attribute, axis=0)
        
        print(f"predicted: {temp.leaf_value}, target: {y}")


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
