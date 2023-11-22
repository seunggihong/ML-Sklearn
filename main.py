import pandas as pd
import matplotlib.pyplot as plt
import Algorithm.DTree as dt
import Algorithm.RF as rf
import Algorithm.NB as nb
import Algorithm.KNN as knn
import Algorithm.Ada as ada
import Algorithm.DA as da
import Algorithm.SVM as svm
import Algorithm.Votting as votting

from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

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
    # gau_nb = nb.gaussian_nb(heart_X, heart_y)
    # print(gau_nb)

    # KNN
    # r_nn = knn.r_knn(wine_X, wine_y)
    # c_nn = knn.c_knn(heart_X, heart_y)
    # print(r_nn)
    # print(c_nn)

    # Ada Boosting
    # r_ad = ada.r_ada(wine_X, wine_y)
    # c_ad = ada.c_ada(heart_X, heart_y)
    # print(r_ad)
    # print(c_ad)

    # Discriminant Analysis
    # liner_da = da.lda(heart_X, heart_y)
    # quad_da = da.qda(heart_X, heart_y)
    # print(liner_da)
    # print(quad_da)

    # SVM
    # reg_svm = svm.r_svm(wine_X, wine_y)
    # class_svm = svm.c_svm(heart_X, heart_y)
    # print(reg_svm)
    # print(class_svm)

    # Votting
    r_vote = votting.r_votting(wine_X, wine_y,
                               est=[
                                   ('SVM', SVR()),
                                   ('KNN', KNeighborsRegressor())
                               ])
    c_vote = votting.c_votting(heart_X, heart_y,
                               est=[
                                   ('SVM', SVC()),
                                   ('KNN', KNeighborsClassifier())
                               ])
    print(r_vote)
    print(c_vote)
