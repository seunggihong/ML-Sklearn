from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from utils.evaluation import model_evaluation


def lda(data, target, params):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = LinearDiscriminantAnalysis()
    grid_search = GridSearchCV(model, param_grid=params)
    grid_search.fit(x_train, y_train)
    y_predict = grid_search.predict(x_test)

    estimator = grid_search.best_estimator_

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate, estimator


def qda(data, target, params):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = QuadraticDiscriminantAnalysis()
    grid_search = GridSearchCV(model, param_grid=params)
    grid_search.fit(x_train, y_train)
    y_predict = grid_search.predict(x_test)

    estimator = grid_search.best_estimator_

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate, estimator
