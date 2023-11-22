from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

from distutils.log import error


def model_evaluation(y_test, y_predict):
    acccuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='weighted')
    recall = recall_score(y_test, y_predict, average='weighted')
    f1 = f1_score(y_test, y_predict, average='weighted')

    return [acccuracy, precision, recall, f1]


def reg_model_evaluation(y_test, y_predict):
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predict)

    return [mse, rmse, r2]


def save_evaluation(path, algo_name_list, model_list):
    try:
        algo_list = algo_name_list
        cols_list = ['accuracy', 'precision', 'recall', 'f1']

        classify_list = model_list
        eval_df = pd.DataFrame(classify_list)
        eval_df = eval_df.values.tolist()
        eval_df = pd.DataFrame(eval_df, columns=cols_list, index=algo_list)

        eval_df.to_csv(path)

        return True
    except error:
        print(error)

        return False
