import pandas as pd

df = pd.read_csv('./dataset/D1.txt',sep=" ", header=None, names=["X1", "X2", "Y"])
for i in range(1, len(df.columns) - 1):
    cname = df.columns[i]
    filtered_df=df.filter([cname,'Y'],axis=1)
    sorted_df=filtered_df.sort_values(cname)
    sorted_df=(sorted_df[0:740])
    print('sorted',sorted_df)
    #print (sorted_df.iloc[[0]])
    #diff_df=sorted_df.diff()
    #print('sorted diff',diff_df)
    #print('diffdf',diff_df.loc[(diff_df['Y'] == 1) | (diff_df['Y'] == -1)])
    #psbl_c_split = diff_df.loc[(diff_df['Y'] == 1) | (diff_df['Y'] == -1)]
    sorted_df['Z'] = sorted_df['Y'].diff(1)
    psbl_c_split_2 = sorted_df.loc[(sorted_df['Z'] == 1) | (sorted_df['Z'] == -1)]
    print(psbl_c_split_2)
