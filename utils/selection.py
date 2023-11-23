from Algorithm.Bagging import r_bagging, c_bagging
from Algorithm.Voting import r_voting, c_voting
from Algorithm.SVM import r_svm, c_svm
from Algorithm.DA import lda, qda
from Algorithm.Ada import r_ada, c_ada
from Algorithm.KNN import r_knn, c_knn
from Algorithm.NB import multinomial_nb, gaussian_nb
from Algorithm.RF import r_random_forest, c_random_forest
from Algorithm.DTree import r_decision_tree, c_decision_tree
import pandas as pd


def wine_data():
    reg_df = pd.read_csv('./data/winequality-red.csv', encoding='utf-8')
    wine_X = reg_df.iloc[:, :11]
    wine_y = reg_df['quality']

    return wine_X, wine_y


def heart_data():
    classify_df = pd.read_csv('./data/heart.csv', encoding='utf-8')
    classify_df = pd.DataFrame(classify_df)

    classify_df = pd.get_dummies(classify_df, dtype='int')

    heart_X = classify_df.drop('HeartDisease', axis=1)
    heart_y = classify_df.pop('HeartDisease')

    return heart_X, heart_y


def select_model(model, problem, parmas, est=None):
    if problem == 'reg':
        data, target = wine_data()
        if model == 'dtree':
            return r_decision_tree(data, target, parmas)
        elif model == 'rf':
            return r_random_forest(data, target, parmas)
        elif model == 'knn':
            return r_knn(data, target, parmas)
        elif model == 'ada':
            return r_ada(data, target, parmas)
        elif model == 'svm':
            return r_svm(data, target, parmas)
        elif model == 'voting':
            return r_voting(data, target, parmas, est)
        elif model == 'bagging':
            return r_bagging(data, target, parmas, est)
        else:
            return 0
    elif problem == 'class':
        data, target = heart_data()
        if model == 'dtree':
            return c_decision_tree(data, target, parmas)
        elif model == 'rf':
            return c_random_forest(data, target, parmas)
        elif model == 'gnb':
            return gaussian_nb(data, target, parmas)
        elif model == 'mnb':
            return multinomial_nb(data, target, parmas)
        elif model == 'knn':
            return c_knn(data, target, parmas)
        elif model == 'ada':
            return c_ada(data, target, parmas)
        elif model == 'lda':
            return lda(data, target, parmas)
        elif model == 'qda':
            return qda(data, target, parmas)
        elif model == 'svm':
            return c_svm(data, target, parmas)
        elif model == 'voting':
            return c_voting(data, target, parmas, est)
        elif model == 'bagging':
            return c_bagging(data, target, parmas, est)
        else:
            return 0
