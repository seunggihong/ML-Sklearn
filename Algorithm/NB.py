from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from utils.evaluation import model_evaluation


def gaussian_nb(data, target, params):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = GaussianNB()
    grid_search = GridSearchCV(model, param_grid=params)
    grid_search.fit(x_train, y_train)
    y_predict = grid_search.predict(x_test)

    estimator = grid_search.best_estimator_

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate, estimator


def multinomial_nb(data, target, params):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = MultinomialNB()
    grid_search = GridSearchCV(model, param_grid=params)
    grid_search.fit(x_train, y_train)
    y_predict = grid_search.predict(x_test)

    estimator = grid_search.best_estimator_

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate, estimator
