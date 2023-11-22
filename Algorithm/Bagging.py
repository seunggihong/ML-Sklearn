from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split
from utils.evaluation import model_evaluation, reg_model_evaluation


def r_bagging(data, target, est):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = BaggingRegressor(estimator=est)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    score = reg_model_evaluation(y_test, y_predict)

    return score


def c_bagging(data, target, est):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = BaggingClassifier(estimator=est)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
