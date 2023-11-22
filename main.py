import pandas as pd
from sklearn.model_selection import train_test_split
import ML.Machine as ml

if __name__ == "__main__":

    # Data load
    df = pd.read_csv('./data/winequality-red.csv', encoding='utf-8')
    wine_df = pd.DataFrame(df)
    wine_df.info()

    X = wine_df.iloc[:, :11]
    y = wine_df['quality']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    # Decision tree
    dtree = ml.decision_tree(x_train, x_test, y_train, y_test)
