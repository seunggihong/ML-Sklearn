from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from utils import model_evaluation, reg_model_evaluation


def r_ada(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = AdaBoostRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    score = reg_model_evaluation(y_test, y_predict)

    return score


def c_ada(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = AdaBoostClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
