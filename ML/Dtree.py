from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split


def r_decision_tree(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    acccuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='weighted')
    recall = recall_score(y_test, y_predict, average='weighted')
    f1 = f1_score(y_test, y_predict, average='weighted')

    return [acccuracy, precision, recall, f1]


def c_decision_tree(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    acccuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='weighted')
    recall = recall_score(y_test, y_predict, average='weighted')
    f1 = f1_score(y_test, y_predict, average='weighted')

    return [acccuracy, precision, recall, f1]
