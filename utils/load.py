import json


def load_params(prob, model_name):
    with open('./hyperparam/params.json') as f:
        params_json = json.load(f)

    return params_json['params'][prob][model_name]
