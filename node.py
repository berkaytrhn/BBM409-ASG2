class Node:
    # Feature column index
    value = None

    # previous node's feature value
    feature_value = None

    # True for leaf nodes, false for other
    is_leaf = None

    # format -> [feature1, feature2], includes Node instances
    children = None

    # not None for leaves, for leaves it specifies prediction    
    leaf_value = None

    def __init__(self, value=None):
        self.value = value
        self.is_leaf = False
        self.children = []
    def __str__(self):
     return f"Feature -> {self.value} -> {self.is_leaf}, {len(self.children)}, {self.leaf_value}"