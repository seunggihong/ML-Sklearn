from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from utils import model_evaluation


def gaussian_nb(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate


def multinomial_nb(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)
    return evaluate
