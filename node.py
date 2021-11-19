from collections import defaultdict


class Node:
    # Feature column index
    value = None

    is_leaf = None

    # [feature1, feature2] 
    children = None

    # not None for leaves    
    leaf_value = None

    def __init__(self, value=None):
        self.value = value
        self.is_leaf = False
        self.children = defaultdict(lambda:False)
    def __str__(self):
     return f"{self.value} attribute -> {self.is_leaf}, {self.children.keys()}, {self.leaf_value}"