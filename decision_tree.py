from math import log
import numpy as np
from numpy.lib.shape_base import column_stack
from node import Node


class DecisionTree:



    def __init__(self):
        self.head = Node()

    def calculate_gains(self, set):
        # fetaure based entropies and information gains
        gains = []
        for column in range(set.shape[1]-1):
            information = 0
            for value in np.unique(set[:,column]):
                # filter for attribute value
                temp_filter = (set[:,column] == value)
                
                # using that filter to get attributes
                temp_samples = set[temp_filter]

                # calculate entropy and number of samples
                temp_entropy, temp_pos, temp_neg = self.calculate_entropy(temp_samples)
                
                # calculate ratio of relevant attribute
                temp_ratio = (temp_pos+temp_neg)/(self.num_pos+self.num_neg)
                
                # summing the result with information
                information += (temp_entropy*temp_ratio)
                #print((temp_entropy*temp_ratio))
            
            gain = self.dataset_entropy-information

            #print(f"For column {column}, gain: {gain}")
            gains.append((column, gain))
        
        gains = sorted(gains, key=lambda x:x[1], reverse=True)
        # print(len(gains))
        return gains[0]


    def fit(self, X, y):
        data = np.concatenate((X,y), axis=1)
        dataset_entropy, num_pos, num_neg = self.calculate_entropy(data)

        # setting attributes
        # features
        self.X = X
        # labels
        self.y = y
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



    def predict(self, X):
        pass 


    def calculate_entropy(self, X):
        positives = X[X[:,-1]==1]
        negatives = X[X[:,-1]==0]

        num_pos = positives.shape[0]
        num_neg = negatives.shape[0]
        num_total = num_pos+num_neg
        
        p_ratio = num_pos/num_total
        n_ratio = num_neg/num_total

        log_p = log(p_ratio, 2) if not p_ratio == 0 else 0
        log_n = log(n_ratio, 2) if not n_ratio == 0 else 0

        entropy = (-p_ratio*log_p)-(n_ratio*log_n)

        return entropy, num_pos, num_neg

    def calculate_gain(self, X, feature=None):
        pass