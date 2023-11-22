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

## Algorithm

- [D-Tree](#dtree)
- [RF](#rf)
- [NB](#nb)
- [K-NN](#knn)
- [Ada](#ada)
- [DA](#da)
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

```python
# Regression
def r_random_forest(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    score = reg_model_evaluation(y_test, y_predict)

    return score
```

```python
# Classification
def c_random_forest(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
```

<hr>

<a name='nb'></a>

### NB

**_Gaussian Naive Bayes(GNB)_**

```python
def gaussian_nb(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate

```

**_Multinomial Naive Bayes(MNB)_**

```python
def multinomial_nb(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)
    return evaluate
```

<hr>

<a name='knn'></a>

### K-NN **_(K Nearest Neighbors)_**

```python
# Regression
def r_knn(data, target, k=3):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    score = reg_model_evaluation(y_test, y_predict)

    return score
```

```python
# Classification
def c_knn(data, target, k=3):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
```

<hr>

<a name='ada'></a>

### Ada **_(Adaptive Boosting)_**

```python
# Regression
def r_ada(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = AdaBoostRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    score = reg_model_evaluation(y_test, y_predict)

    return score
```

```python
# Classification
def c_ada(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = AdaBoostClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
```

<hr>

<a name='da'></a>

### DA **_(Discriminant Analysis)_**

**_Linear Discriminant Analysis(LDA)_**

```python
def lda(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = LinearDiscriminantAnalysis()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
```

**_Quadratic Discriminant Analysis(QDA)_**

```python
def qda(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
```

<hr>

<a name='svm'></a>

### SVM **_(Support Vector Machine)_**

```python
# Regression
def r_svm(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = SVR(C=10, epsilon=0.2)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    score = reg_model_evaluation(y_test, y_predict)

    return score
```

```python
# Classification
def c_svm(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = SVC(C=10, kernel='linear')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
```

<hr>

<a name='voting'></a>

### Voting

```python
# Regression
def r_votting(data, target, est, jobs=1):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = VotingRegressor(estimators=est, n_jobs=jobs)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    score = reg_model_evaluation(y_test, y_predict)

    return score
```

```python
# Classification
def c_votting(data, target, est, voting='hard'):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = VotingClassifier(estimators=est, voting=voting)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
```

<hr>

<a name='bagging'></a>

### Bagging

```python
# Regression
def r_bagging(data, target, est):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = BaggingRegressor(estimator=est)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    score = reg_model_evaluation(y_test, y_predict)

    return score
```

```python
# Classification
def c_bagging(data, target, est):
    x_train, x_test, y_train, y_test = train_test_split(data, target)

    model = BaggingClassifier(estimator=est)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    evaluate = model_evaluation(y_test, y_predict)

    return evaluate
```

<hr>

## Reference

- <a href='https://scikit-learn.org/stable/user_guide.html'>scikit-learn docs</a>
