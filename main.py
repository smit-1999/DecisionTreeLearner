import pandas as pd
import time
from sklearn import metrics
from Node import Node
from draw_boundary import draw
from infogain import calculate_infogain
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def split(df, c_name, threshold):
    ''' function to split the data '''

    # left = np.array(
    #     [row for row in df if row[feature_index] <= threshold])
    # right = np.array(
    #     [row for row in df if row[feature_index] > threshold])
    left = df.loc[df[c_name] <= threshold]
    right = df.loc[df[c_name] > threshold]
    return left, right


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


def rec(df, parent):
    psbl_split = calculate_psbl_splits(df)
    max_info_gain, max_threshold, max_key, left_data, right_data = best_split(
        psbl_split, df)
    if len(left_data) > 0 and len(right_data) > 0 and max_info_gain != "null" and max_info_gain > 0:
        newNode = Node(max_key, max_threshold,
                       [], [], max_info_gain, parent=parent)
        # recur left
        left_subtree = rec(left_data, newNode)
        # recur right
        right_subtree = rec(right_data, newNode)
        newNode.leftChild = left_subtree
        newNode.rightChild = right_subtree
        return newNode

    leaf_value = calculate_leaf_value(df['Y'])
    return Node(value=leaf_value, parent=parent)


def predict(X, root):
    predictions = []
    for i in range(0, (X.shape[0])):
        predictions.append(make_prediction(X.iloc[i, :], root))
    return predictions


def make_prediction(x, curr):
    if curr.value != None:
        return curr.value
    # print('make rpediction', x, type(x))
    feature_val = x[curr.index]
    if feature_val <= curr.threshold:
        return make_prediction(x, curr.leftChild)
    else:
        return make_prediction(x, curr.rightChild)


def train(df):
    feature_cols = ['X1', 'X2']
    X = df[feature_cols]
    Y = df['Y']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=.1808)
    dataset_size = [32, 128, 512, 2048, 8192, 12000]
    error_rate = []
    for i in range(0, len(dataset_size)):
        sz = min(dataset_size[i], len(X))
        x_train_subset = X_train.iloc[:sz]
        y_train_subset = pd.DataFrame(Y_train.iloc[:sz])
        x_train_subset['Y'] = y_train_subset
        root = rec(x_train_subset, None)
        Y_pred = predict(X_test, root)
        error_rate.append(1 - metrics.accuracy_score(Y_test, Y_pred))
        draw(root, sz)
    plt.plot(dataset_size, error_rate)
    plt.xlabel('Number of Points n')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.show()


def main():
    start = time.time()
    df = pd.read_csv('./dataset/Dbig.txt', sep=" ",
                     header=None, names=["X1", "X2", "Y"])
    # root = rec(df, None)
    train(df)
    end = time.time()
    print('Time elapsed ', end-start)
    # print_tree(root)
    # print_tree_bfs(root)


main()
