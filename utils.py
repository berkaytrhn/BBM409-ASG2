import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def display_confusion_matrix(confusion_matrix):
    # heatmap display for confusion matrix
    labels = ["True Neg","False Pos","False Neg","True Pos"]
    length = len(max(labels))+4
    labels = np.asarray(labels).reshape(2,2)
    
    annots = [f"{str(label)}({str(value)})" for array in np.dstack((labels,confusion_matrix)) for (label, value) in array]
    annots = np.asarray(annots).reshape(2,2).astype(str)
    plt.figure(figsize = (10,7))
    sns.heatmap(confusion_matrix, annot=annots, cmap="YlGnBu", fmt=f".{length}")
    plt.show()




def create_column_map(columns):
    # create dictionary as {"column_index": "column_name", ....}
    return {index:columns[index] for index in range(len(list(columns))-1)}

def k_fold_cross_validation(data, split):
    # function for 5 fold cross validation splitter, returns splitted data 
    kfold = KFold(n_splits=split, shuffle=True,random_state=123456)


    splitted = []
    for _train_index, _test_index in kfold.split(data):
        train, test = data[_train_index], data[_test_index]
        print(train.shape)
        print(test.shape)
        splitted.append((train, test))

    return splitted

    
    


def encode_age(column, max, min):
    # discretizing age as 0 and 1
    _range = ((max-min)/2)
    
    intervals = []

    for index in range(1,3):
        temp_max = min+_range

        intervals.append((int(min), int(temp_max)))
        min = temp_max

    encoded = []
    for age in column:
        for index, interval in enumerate(intervals):
            __range = range(interval[0],interval[1]+1)
            if age in __range:
                encoded.append(index)
                break
    
    return encoded
        

def encode_features(data):
    """
    encoding all features as 0 and 1(also calls encode_age function)
    """


    # get columns which includes yes or no as a value
    binary_columns = data.columns.tolist()
    binary_columns.remove("Age")

    # Appling binary encoding to 'Age' column
    ages = data["Age"].tolist()
    max_age = max(ages)
    min_age = min(ages)
    

    encoded_ages = encode_age(ages, max_age, min_age)
  
    # replace age column with encoded version
    data["Age"] = encoded_ages

    for col in binary_columns:
        values_array = np.unique(data[col].tolist())
        if not (("Yes" in values_array[0]) or ("Positive" in values_array[0])):
            # little configuration for giving 1 for Yes and 0 for No 
            values_array = values_array[::-1]
        
        data[col] = data[col].apply(lambda x:1 if x==values_array[0] else 0)

    
    return data 
