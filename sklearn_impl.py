import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


def sklearn():
    df = pd.read_csv('./dataset/Dbig.txt', sep=" ",
                     header=None, names=["X1", "X2", "Y"])
    feature_cols = ['X1', 'X2']
    X = df[feature_cols]  # Features
    Y = df['Y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1808, random_state=1)
    print(len(X_train), len(X_test))
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print("Error rate:", 1 - metrics.accuracy_score(y_test, y_pred))

    dataset_size = [32, 128, 512, 2048, 8192]
    error_rate = []
    for i in range(0, len(dataset_size)):
        sz = dataset_size[i]
        x_train_subset = X_train.iloc[:sz]
        y_train_subset = y_train.iloc[:sz]
        clf = DecisionTreeClassifier()
        clf = clf.fit(x_train_subset, y_train_subset)
        y_pred_subset = clf.predict(X_test)
        # print(x_train_subset.shape, y_train_subset.shape, y_pred_subset.shape)
        # print("Error rate:", 1 - metrics.accuracy_score(y_test, y_pred_subset))
        error_rate.append(1 - metrics.accuracy_score(y_test, y_pred_subset))

    plt.plot(dataset_size, error_rate)
    plt.show()


sklearn()
