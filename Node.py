class Node():
    def __init__(self, index=None, threshold=None, leftChild=None, rightChild=None, info_gain=None, value=None):
        self.index = index
        self.threshold = threshold
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.info_gain = info_gain

        self.value = value
