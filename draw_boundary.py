import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Callable


def draw_decision_boundary(model_function: Callable, grid_abs_bound: float = 1.0, savefile: str = None):
    # """`model_function` should be your model's formula for evaluating your decision tree, returning either `0` or `1`.
    # \n`grid_abs_bound` represents the generated grids absolute value over the x-axis, default value generates 50 x 50 grid.
    #  \nUse `grid_abs_bound = 1.0` for question 6 and `grid_abs_bound = 1.5` for question 7.
    #   \nSet `savefile = 'plot-save-name.png'` to save the resulting plot, adjust colors and scale as needed."""

    colors = ['#91678f', '#afd6d2']  # hex color for [y=0, y=1]
    xval = np.linspace(grid_abs_bound, -grid_abs_bound,
                       50).tolist()  # grid generation
    xdata = []
    for i in range(len(xval)):
        for j in range(len(xval)):
            xdata.append([xval[i], xval[j]])

    # creates a dataframe to standardize labels
    df = pd.DataFrame(data=xdata, columns=['x_1', 'x_2'])
    # applies model from model_function arg
    df['y'] = df.apply(model_function, axis=1)
    d_columns = df.columns.to_list()  # grabs column headers
    y_label = d_columns[-1]  # uses last header as label
    d_xfeature = d_columns[0]  # uses first header as x_1 feature
    d_yfeature = d_columns[1]  # uses second header as x_1 feature
    # sorts by label to ensure correct ordering in plotting loop
    df = df.sort_values(by=y_label)

    d_xlabel = f"feature  $\mathit{{{d_xfeature}}}$"  # label for x-axis
    dy_ylabel = f"feature  $\mathit{{{d_yfeature}}}$"  # label for y-axis
    plt.xlabel(d_xlabel, fontsize=10)  # set x-axis label
    plt.ylabel(dy_ylabel, fontsize=10)  # set y-axis label
    legend_labels = []  # create container for legend labels to ensure correct ordering

    # loop through placeholder dataframe
    for i, label in enumerate(df[y_label].unique().tolist()):
        df_set = df[df[y_label] == label]  # sort according to label
        set_x = df_set[d_xfeature]  # grab x_1 feature set
        set_y = df_set[d_yfeature]  # grab x_2 feature set
        # marker='s' for square, s=40 for size of squares large enough
        plt.scatter(set_x, set_y, c=colors[i], marker='s', s=40)
        legend_labels.append(
            f"""{y_label} = {label}""")  # apply labels for legend in the same order as sorted dataframe

    plt.title("Model Decision Boundary Example", fontsize=12)  # set plot title
    ax = plt.gca()  # grab to set background color of plot
    # set aforementioned background color in hex color
    ax.set_facecolor('#2b2d2e')
    plt.legend(legend_labels)  # create legend with sorted labels

    if savefile is not None:  # save your plot as .png file
        plt.savefig(savefile)
    plt.show()  # show plot with decision bounds


def model_y(row):
    """example model used to demonstrate drawing decision bounds for hw2"""
    x_1, x_2 = row.x_1, row.x_2  # grabs standardized labels from pandas.apply function input and renames to more familiar variables
    if x_1 >= 0.0:
        if x_2 >= 0.0:
            return 0
        return 1
    if x_2 >= 0.0:
        return 1
    return 0


# generate decision boundary plot
draw_decision_boundary(model_function=model_y, grid_abs_bound=1)
