import collections


def print_tree(curr=None):

    if curr.value is not None:
        print('Leaf node', curr.value,
              '' if curr.parent is None else 'Parent threshold' + str(curr.parent.threshold))
    else:
        print(str(curr.index), "<=",
              curr.threshold, "?", curr.info_gain, '' if curr.parent is None else 'Parent threshold' + str(curr.parent.threshold))
    # print("%sleft:")
    if curr.leftChild is not None:
        print_tree(curr.leftChild)
    # print("%sright:" % (indent), end="")
    if curr.rightChild is not None:
        print_tree(curr.rightChild)


def print_tree_bfs(curr):
    ans = []

    if curr is None:
        return ans

    # Initialize queue
    queue = collections.deque()
    queue.append(curr)

    # Iterate over the queue until it's empty
    while queue:
        currSize = len(queue)
        currList = []

        while currSize > 0:
            currNode = queue.popleft()
            currList.append(
                {currNode.index, currNode.threshold, currNode.info_gain})
            currSize -= 1

            # Check for left child
            if currNode.leftChild is not None:
                queue.append(currNode.leftChild)
            # Check for right child
            if currNode.rightChild is not None:
                queue.append(currNode.rightChild)

        # Append the currList to answer after each iteration
        ans.append(currList)

    for row in ans:
        print(row)
        print("\n\n")
