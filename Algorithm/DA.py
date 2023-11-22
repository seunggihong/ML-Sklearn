from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from utils.evaluation import model_evaluation, reg_model_evaluation


def lda(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = LinearDiscriminantAnalysis()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate


def qda(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
