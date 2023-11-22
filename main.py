import pandas as pd
import matplotlib.pyplot as plt
import ML.DTree as dt
import ML.RF as rf
import ML.NB as nb


if __name__ == "__main__":
    # Heart failure prediction data (Classification)
    classify_df = pd.read_csv('./data/heart.csv', encoding='utf-8')
    classify_df = pd.DataFrame(classify_df)

    classify_df = pd.get_dummies(classify_df, dtype='int')

    heart_X = classify_df.drop('HeartDisease', axis=1)
    heart_y = classify_df.pop('HeartDisease')

    # Red wine quality data (Regression)
    reg_df = pd.read_csv('./data/winequality-red.csv', encoding='utf-8')
    wine_X = reg_df.iloc[:, :11]
    wine_y = reg_df['quality']

    # ML model

    # Decision Tree
    # c_dt = dt.c_decision_tree(heart_X, heart_y)
    # r_dt = dt.r_decision_tree(wine_X, wine_y)
    # print(c_dt)
    # print(r_dt)

    # labels = ['acccuracy', 'precision', 'recall', 'f1']
    # plt.scatter(labels, c_dt, c='red')
    # plt.scatter(labels, r_dt, c='gray')
    # plt.show()

    # Random Forest
    # c_rf = rf.c_random_forest(heart_X, heart_y)
    # r_rf = rf.r_random_forest(wine_X, wine_y)
    # print(c_rf)
    # print(r_rf)

    # NaiveBayes
    gau_nb = nb.gaussian_nb(heart_X, heart_y)
    print(gau_nb)
