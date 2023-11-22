from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def decision_tree(x_train, x_test, y_train, y_test):
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(x_train, y_train)
    y_predict = decision_tree.predict(x_test)

    acccuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='weighted')
    recall = recall_score(y_test, y_predict, average='weighted')
    f1 = f1_score(y_test, y_predict, average='weighted')
    return [acccuracy, precision, recall, f1]
