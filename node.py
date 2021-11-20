from collections import defaultdict


class Node:
    # Feature column index
    value = None

    # previous node's feature value
    feature_value = None

    is_leaf = None

    # gain for this branch from previous 
    gain = None

    # [feature1, feature2] 
    children = None

    # not None for leaves    
    leaf_value = None

    def __init__(self, value=None):
        self.value = value
        self.is_leaf = False
        self.children = []
    def __str__(self):
     return f"{self.value} attribute -> {self.is_leaf}, {len(self.children)}, {self.leaf_value}"