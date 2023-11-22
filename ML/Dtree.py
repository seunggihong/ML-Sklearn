from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from utils import model_evaluation


def r_decision_tree(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate


def c_decision_tree(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
