def print_tree(tree=None):
    ''' function to print the tree '''

    # if not tree:
    #     tree = root

    if tree.value is not None:
        print('Leaf node', tree.value)
    else:
        print(str(tree.index), "<=",
              tree.threshold, "?", tree.info_gain)
    # print("%sleft:")
    if tree.leftChild is not None:
        print_tree(tree.leftChild)
    # print("%sright:" % (indent), end="")
    if tree.rightChild is not None:
        print_tree(tree.rightChild)
