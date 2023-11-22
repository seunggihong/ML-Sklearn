from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils import model_evaluation


def r_random_forest(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    # y_predict = model.predict(x_test)

    evaluate = model.score(x_test, y_test)

    return evaluate


def c_random_forest(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
