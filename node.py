class Node:
    # None for normal, value if leaf
    value = None
    children = None

    def __init__(self, value=None):
        self.value = value
        self.children = []
