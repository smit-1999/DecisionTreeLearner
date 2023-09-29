
from entropy import calculate_entropy
import math


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
    left_entropy_split = 0
    right_entropy_split = 0
    if w_l == 0:
        left_entropy_split = 0
    else:
        left_entropy_split = w_l*math.log2(w_l)
    if w_r == 0:
        right_entropy_split = 0
    else:
        right_entropy_split = w_r*math.log2(w_r)
    entropy_split = -(left_entropy_split) - (right_entropy_split)
    gain_ratio = info_gain/entropy_split
    return gain_ratio
