import pandas as pd
import numpy as np
from decision_tree import DecisionTree

def encode_age(column, max, min):
    _range = ((max-min)/5)
    
    intervals = []

    for index in range(1,6):
        temp_max = min+_range

        intervals.append((int(min), int(temp_max)))
        min = temp_max
        #min = temp_max

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


def ID3(root, examples, attributes):

    # NODE EKLENDİĞİNDE EKLENENİ ROOT YAP

    if labelStatusCheck(examples) == 1:
        # tree change
        return root
    elif labelStatusCheck(examples) == -1:
        # tree change
        return root
    elif attributes.size() == 0:
        # mostCommonTargetAttribute(examples) operations
        return root
    else:
        a = root.calculate_gains(examples)[0]
        #node creation for a and assigning a to the root node
        # each value for A (yes or no / 1 2 3 4 5), add branch ro root
        # yes/no içeren tüm subset sampleları setle (binary için sette iki liste olacak)
        # -> setlenen şey boş işe altına branch ile leaf ekle ve mostCommonTargetAttribute çağır (setlenen ile değil bir üstüyle)
        # -> else, ID3(root,set,attributes-A)


def labelStatusCheck(examples):
    posCount = None
    negCount = None

    for i in examples[:, -1].tolist():
        if i == 1:
            posCount = posCount + 1
        else:
            negCount = negCount + 1

    if posCount != 0 and negCount == 0:
        return 1
    elif negCount != 0 and posCount == 0:
        return -1
    else:
        return 0

def mostCommonTargetAttribute(examples):
    posCount = None
    negCount = None

    for i in examples[:, -1].tolist():
        if i == 1:
            posCount = posCount + 1
        else:
            negCount = negCount + 1

    return max(posCount,negCount)

def main():

    data = pd.read_csv("diabetes_data_upload.csv")

    data = encode_features(data)   

    #data.to_csv("test.csv",index=False)
    
    tree = DecisionTree()
    
    data = np.array(data)
    
    
    X = data[:,:-1]
    y = data[:,-1].reshape(-1,1)
    tree.fit(X, y)
    


if __name__ == "__main__":
    main()
