import pandas as pd
import math


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

# once we have the possible splits we will try to find info gain for the same
# h_y = -p0*logp0 - p1*logp1
# where p0=countOfZeros/totalValues, p1=countOfOnes/totalValues
c0 = df['Y'].value_counts()[0]
c1 = df['Y'].value_counts()[1]
c_total = c0 + c1

p0 = c0/c_total
log_2_p0 = math.log2(p0)

p1 = c1/c_total
log_2_p1 = math.log2(p1)

h_y = (-p0*log_2_p0) - (p1*log_2_p1)
