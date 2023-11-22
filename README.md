![Static Badge](https://img.shields.io/badge/python3-3.11.5-%233776AB?style=plastic&logo=python&logoColor=white)
![Static Badge](https://img.shields.io/badge/sklearn-1.3.2-%23F7931E?style=plastic&logo=scikitlearn&logoColor=white)
![Static Badge](https://img.shields.io/badge/pandas-2.1.2-%23150458?style=plastic&logo=scikitlearn&logoColor=white)

# ML-Sklearn

Study for Scikit-learn Machine learing

#

## Data

![Static Badge](https://img.shields.io/badge/kaggle-Red%20Wine%20Quality-%2320BEFF?style=social&logoColor=white&labelColor=%2320BEFF)

![Static Badge](https://img.shields.io/badge/kaggle-Heart%20Failure%20Prediction-%2320BEFF?style=social&logoColor=white&labelColor=%2320BEFF)

- Regression

  - <a href='https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009'>Red Wine Quality</a>

- Classification

  - <a href='https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction'>Heart Failure Prediction</a>

<hr>

## Machine

- [D-Tree](#dtree)
- [RF](#rf)
- [GNB](#gnb)
- [K-NN](#knn)
- [Ada](#ada)
- [QDA](#qda)
- [SVM](#svm)
- [Voting](#voting)
- [Bagging](#bagging)

<hr>

<a name='dtree'></a>

### D-Tree **_(Decision Tree)_**

```python
# Regression
def r_decision_tree(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    score = reg_model_evaluation(y_test, y_predict)

    return score
```

```python
# Classification
def c_decision_tree(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
```

<hr>
<a name='rf'></a>

### RF **_(Random Forest)_**

```bash

```

<hr>

<a name='gnb'></a>

### GNB

```bash

```

<hr>

<a name='knn'></a>

### K-NN

```bash

```

<hr>

<a name='ada'></a>

### Ada

```bash

```

<hr>

<a name='qda'></a>

### QDA

```bash

```

<hr>

<a name='svm'></a>

### SVM

```bash

```

<hr>

<a name='voting'></a>

### Voting

```bash

```

<hr>

<a name='bagging'></a>

### Bagging

```bash

```

<hr>
