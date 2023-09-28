import pandas as pd
import math
import time
from Node import Node
from print_tree import print_tree
from entropy import calculate_entropy


def split(df, c_name, threshold):
    ''' function to split the data '''

    # left = np.array(
    #     [row for row in df if row[feature_index] <= threshold])
    # right = np.array(
    #     [row for row in df if row[feature_index] > threshold])
    left = df.loc[df[c_name] <= threshold]
    right = df.loc[df[c_name] > threshold]
    return left, right


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


def calculate_psbl_splits(df):
    psbl_split = {}
    for i in range(0, len(df.columns) - 1):
        cname = df.columns[i]
        filtered_df = df.filter([cname, 'Y'], axis=1)
        sorted_df = filtered_df.sort_values(cname)
        # diff_df=sorted_df.diff()
        # psbl_c_split = diff_df.loc[(diff_df['Y'] == 1) | (diff_df['Y'] == -1)]
        sorted_df['Z'] = sorted_df['Y'].diff(1)
        diff_Z = sorted_df.loc[(
            sorted_df['Z'] == 1) | (sorted_df['Z'] == -1)]
        # tst = sorted_df.loc[(sorted_df['Z'] == 1) | (
        #     sorted_df['Z'] == -1), cname]
        if cname in psbl_split.keys():
            psbl_split[cname].append(diff_Z[cname].values.flatten())
        else:
            psbl_split[cname] = diff_Z[cname].values.flatten()
    return psbl_split


def best_split(psbl_split, df):
    max_info_gain = 0
    max_key = 0
    max_threshold = 0
    left_data = []
    right_data = []
    for key in (psbl_split.keys()):
        val = psbl_split[key]
        for threshold in val:
            left, right = split(df, key, threshold)
            info_gain = calculate_infogain(df, left, right)
            if info_gain == 'null':
                continue
            if info_gain >= max_info_gain:
                max_info_gain = info_gain
                max_threshold = threshold
                max_key = key
                left_data = left
                right_data = right

    return max_info_gain, max_threshold, max_key, left_data, right_data


def calculate_leaf_value(Y):
    Y = list(Y)
    return max(Y, key=Y.count)


def rec(df):
    psbl_split = calculate_psbl_splits(df)
    max_info_gain, max_threshold, max_key, left_data, right_data = best_split(
        psbl_split, df)
    if len(left_data) > 0 and len(right_data) > 0 and max_info_gain != "null" and max_info_gain > 0:
        # recur left
        left_subtree = rec(left_data)
        # recur right
        right_subtree = rec(right_data)
        # return decision node
        return Node(max_key, max_threshold,
                    left_subtree, right_subtree, max_info_gain)

    leaf_value = calculate_leaf_value(df['Y'])
    # return leaf node
    return Node(value=leaf_value)


def main():
    start = time.time()
    df = pd.read_csv('./dataset/D2.txt', sep=" ",
                     header=None, names=["X1", "X2", "Y"])
    root = rec(df)
    end = time.time()
    print('Time elapsed ', end-start)
    print_tree(root)


main()
