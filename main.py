import pandas as pd
import numpy as np
from decision_tree import DecisionTree
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def encode_age(column, max, min):
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



def test(tree, X, y):
    # testing tree
    counter = 0
    correct = 0
    for index in range(len(X)):
        
        X_row = X[index]
        y_row = y[index]
        res = tree.predict(X_row, y_row) 
        if res:
            correct += 1
        counter += 1
        print(f"Current Accuracy: {round(correct/counter*100, 4)}%")

    print(f"Accuracy: {round(correct/counter*100, 4)}%")

def main():

    _file =  open("test.txt", "w")
    sys.stdout = _file


    data = pd.read_csv("diabetes_data_upload.csv")

    data = encode_features(data)   

    columns = data.columns
    column_map = {index:columns[index] for index in range(len(list(columns))-1)}
    
    
    data.to_csv("test.csv",index=False)
    
    tree = DecisionTree()
    
    data = np.array(data)

    kfold = KFold(n_splits=5, shuffle=True)
    print("train_test_split shapes:")
    for _train, _test in kfold.split(data):
        print("----------")
        print(_train.shape, _test.shape)

    X = data[:,:-1]
    y = data[:,-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    
    X = X_train
    y = y_train.reshape(-1,1)
    tree.fit(X, y)
    
    test(tree, X_test, y_test)

    print("eqwlekqmekqwem")
    tree.display_tree(tree.head, column_map)

    _file.close()


if __name__ == "__main__":
    main()
