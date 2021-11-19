from collections import defaultdict


class Node:
    # Feature column index
    value = None
    

    is_leaf = None

    # [feature1, feature2] 
    children = None

    # not None for leaves    
    out_class = None

    def __init__(self, value=None):
        self.value = value
        self.is_leaf = False
        self.children = defaultdict(lambda:False)
