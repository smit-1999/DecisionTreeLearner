import pandas as pd
import matplotlib.pyplot as plt

# Use scatter plot for q2.6 D1.txt, D2.txt


def scatter_plot():
    df = pd.read_csv('./dataset/D1.txt', sep=" ",
                     header=None, names=["X1", "X2", "Y"])
    plt.scatter(df["X1"], df["X2"], c=df['Y'])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend
    plt.show()


scatter_plot()
