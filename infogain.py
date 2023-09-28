
from entropy import calculate_entropy


def calculate_infogain(df, left, right):
    initial_entropy = calculate_entropy(df)
    w_l = len(left)/len(df)
    w_r = len(right)/len(df)
    left_entropy = calculate_entropy(left)
    # if (left_entropy == 'null'):
    # return 'null'
    right_entropy = calculate_entropy(right)
    # if (right_entropy == 'null'):
    #    return 'null'
    final_entropy = (w_l*left_entropy) + (w_r*right_entropy)
    info_gain = initial_entropy-final_entropy
    return info_gain
