from math import log
import numpy as np
from numpy.lib.shape_base import column_stack
import pandas as pd
from node import Node


class DecisionTree:



    def __init__(self):
        self.head = Node()

    def calculate_gains(self, _set):
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
                temp_entropy, temp_pos, temp_neg = self.calculate_entropy(temp_samples)
                
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


    def ID3(self, root, examples, attributes):
        print(f"root: ", root)
        # NODE EKLENDİĞİNDE EKLENENİ ROOT YAP

        if self.labelStatusCheck(examples) == 1:
            # tree change, all positive
            root.is_leaf = True
            root.leaf_value = 1
            print("all 1 leaf ->",root)
            return 
        elif self.labelStatusCheck(examples) == 0: 
            # tree change, all negative
            root.is_leaf = True
            root.leaf_value = 0
            print("all 0 leaf ->",root)
            return 
        elif attributes.size == 0:
            # mostCommonTargetAttribute(examples) operations
            print(f"examples: {examples}")
            res = self.mostCommonTargetAttribute(examples)

            root.is_leaf = True
            root.leaf_value = res
            print("most common leaf ->",root)
            return 
        else:
            a = self.calculate_gains(examples)[0]
            
            values = np.unique(examples[:,a])

            
            print("**********************************")
            print(f"Selected feature: {a}")
            print(f"Shape: {examples.shape}")
            for key in values:
                
                """
                {
                    '0': filtered1,
                    '1': filtered2
                }
                """
                _filter = examples[:,a] == key
                new_examples = examples[_filter]
                
                #temp_node.children[key] = new_examples
                

                print(f"{a} -> For {key}")#: {new_examples}")
                res = self.labelStatusCheck(new_examples)
                if not res == -1:
                    leaf_node = Node()
                    root.value = a
                    leaf_node.is_leaf = True
                    leaf_node.leaf_value = res
                    root.children[key] = leaf_node
                    print("classic leaf ->",leaf_node)
                    return 
                else:
                    # remove prvious selected most gained column
                    temp_node = Node()
                    root.value = a
                    root.children[key] = temp_node
                    new_examples = np.concatenate((examples[:,:a],examples[:,a+1:]), axis=1)
                    self.ID3(temp_node,new_examples,self.get_columns(new_examples))
                    print("standard -> ",temp_node)
                    #print(f"Removed -> {new_examples}")

                    

            

            #node creation for a and assigning a to the root node
            # each value for A (yes or no / 1 2 3 4 5), add branch to root
            # yes/no içeren tüm subset sampleları setle (binary için sette iki liste olacak)
            # -> setlenen şey boş ise altına branch ile leaf ekle ve mostCommonTargetAttribute çağır (setlenen ile değil bir üstüyle)
            # -> else, ID3(root,set,attributes-A)


    def get_columns(self, data):
        return np.array(list(range(len(data[0,:-1]))))

    def fit(self, X, y):
        data = np.concatenate((X,y), axis=1)
        #dataset_entropy, num_pos, num_neg = self.calculate_entropy(data)

        # setting attributes
        # features
        self.X = X
        # labels
        self.y = y


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