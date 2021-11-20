import pandas as pd
import numpy as np
from decision_tree import DecisionTree
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



def cross_validation(splitted, column_map):
    counter = 0
    for row in splitted:
        _train, _test = row

        counter += 1
        X_train = _train[:,:-1]
        y_train = _train[:,-1].reshape(-1,1)

        X_test = _test[:,:-1]
        y_test = _test[:,-1].reshape(-1,1)


        tree = DecisionTree()
        
    
        tree.fit(X_train, y_train)

        tree.display_tree(tree.head, column_map)
        predictions, label = tree.predict(X_test, y_test)

        
        precision = precision_score(label, predictions)
        recall = recall_score(label, predictions)
        f1 = f1_score(label, predictions) 
        accuracy = accuracy_score(label, predictions)
        _confusion_matrix = confusion_matrix(label, predictions)
        display_confusion_matrix(_confusion_matrix)
        print(f"{counter} -> Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}")


def main():



    data = pd.read_csv("diabetes_data_upload.csv")


    data = encode_features(data)   

    column_map = create_column_map(data.columns)
       
    
    data = np.array(data)


    splitted = k_fold_cross_validation(data, 5)
    
    cross_validation(splitted, column_map)    
    



if __name__ == "__main__":
    main()
