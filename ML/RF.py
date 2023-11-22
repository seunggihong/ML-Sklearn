from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from utils import model_evaluation


def r_random_forest(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    evaluate = model_evaluation()

    return evaluate


def c_random_forest(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    evaluate = model_evaluation()

    return evaluate
