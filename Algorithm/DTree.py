from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from utils.evaluation import model_evaluation, reg_model_evaluation


def r_decision_tree(data, target, params):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = DecisionTreeRegressor()
    grid_search = GridSearchCV(model, param_grid=params)
    grid_search.fit(x_train, y_train)
    y_predict = grid_search.predict(x_test)

    estimator = grid_search.best_estimator_

    score = reg_model_evaluation(y_test, y_predict)

    return score, estimator


def c_decision_tree(data, target, params):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(model, param_grid=params)
    grid_search.fit(x_train, y_train)
    y_predict = grid_search.predict(x_test)

    estimator = grid_search.best_estimator_

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate, estimator
