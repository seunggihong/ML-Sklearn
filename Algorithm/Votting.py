from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import train_test_split
from utils.evaluation import model_evaluation, reg_model_evaluation


def r_votting(data, target, est, jobs=1):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = VotingRegressor(estimators=est, n_jobs=jobs)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    score = reg_model_evaluation(y_test, y_predict)

    return score


def c_votting(data, target, est, voting='hard'):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = VotingClassifier(estimators=est, voting=voting)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
