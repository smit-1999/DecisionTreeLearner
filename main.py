import pandas as pd
import math
import numpy as np


def calculate_entropy(df):
    # once we have the possible splits we will try to find entropy for the same
    # h_y = -p0*logp0 - p1*logp1
    # where p0=countOfZeros/totalValues, p1=countOfOnes/totalValues
    print('valuecounts', df['Y'].value_counts())
    c0 = 0
    c1 = 0
    if df['Y'].value_counts().get(0, 0):
        c0 = df['Y'].value_counts()[0]
    if df['Y'].value_counts().get(1, 0):
        c1 = df['Y'].value_counts()[1]
    print('c0', c0, 'c1', c1)
    if c0 == 0:
        return 'null'
    if c1 == 0:
        return 'null'
    c_total = c0 + c1

    p0 = c0/c_total
    log_2_p0 = math.log2(p0)

    p1 = c1/c_total
    log_2_p1 = math.log2(p1)

    h_y = (-p0*log_2_p0) - (p1*log_2_p1)
    return h_y


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
    print('calculating infgain')
    initial_entropy = calculate_entropy(df)
    print('initial,', initial_entropy)
    w_l = len(left)/len(df)
    w_r = len(right)/len(df)
    print('weights left,right', w_l, w_r)
    left_entropy = calculate_entropy(left)
    if (left_entropy == 'null'):
        return 'null'
    print('leftentropy', left_entropy)
    right_entropy = calculate_entropy(right)
    if (right_entropy == 'null'):
        return 'null'
    print('rightnetropt', right_entropy)
    final_entropy = (w_l*left_entropy) + (w_r*right_entropy)
    info_gain = initial_entropy-final_entropy
    return info_gain


df = pd.read_csv('./dataset/D1.txt', sep=" ",
                 header=None, names=["X1", "X2", "Y"])
psbl_split = {}
for i in range(0, len(df.columns) - 1):
    cname = df.columns[i]
    filtered_df = df.filter([cname, 'Y'], axis=1)
    sorted_df = filtered_df.sort_values(cname)
    # sorted_df = (sorted_df[0:177])
    # diff_df=sorted_df.diff()
    # print('diffdf',diff_df.loc[(diff_df['Y'] == 1) | (diff_df['Y'] == -1)])
    # psbl_c_split = diff_df.loc[(diff_df['Y'] == 1) | (diff_df['Y'] == -1)]
    sorted_df['Z'] = sorted_df['Y'].diff(1)
    diff_Z = sorted_df.loc[(
        sorted_df['Z'] == 1) | (sorted_df['Z'] == -1)]
    # tst = sorted_df.loc[(sorted_df['Z'] == 1) | (
    #     sorted_df['Z'] == -1), cname]
    # print('tst', tst)
    if cname in psbl_split.keys():
        psbl_split[cname].append(diff_Z[cname].values.flatten())
    else:
        psbl_split[cname] = diff_Z[cname].values.flatten()


curr_info_gain = 0
max_info_gain = curr_info_gain
print('All keys', psbl_split.keys())
for key in (psbl_split.keys()):
    print('Key', key)
    val = psbl_split[key]
    print('Vl', val)
    for threshold in val:
        # threshold = val[0]
        print('In loop', key, threshold)
        left, right = split(df, key, threshold)
        print('leftcount', len(left), 'rightcount', len(right))
        info_gain = calculate_infogain(df, left, right)
        if info_gain == 'null':
            continue
        print('infogain competed', info_gain)
        if info_gain >= max_info_gain:
            max_info_gain = info_gain
            max_threshold = threshold
            max_key = key
