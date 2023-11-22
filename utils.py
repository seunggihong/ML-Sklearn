from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def model_evaluation(y_test, y_predict):
    acccuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='weighted')
    recall = recall_score(y_test, y_predict, average='weighted')
    f1 = f1_score(y_test, y_predict, average='weighted')
    return [acccuracy, precision, recall, f1]
