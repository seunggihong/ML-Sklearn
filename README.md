![Static Badge](https://img.shields.io/badge/python3-3.11.5-%233776AB?style=plastic&logo=python&logoColor=white)
![Static Badge](https://img.shields.io/badge/sklearn-1.3.2-%23F7931E?style=plastic&logo=scikitlearn&logoColor=white)
![Static Badge](https://img.shields.io/badge/pandas-2.1.2-%23150458?style=plastic&logo=scikitlearn&logoColor=white)

## ML-Sklearn

This repository uses scikit-learn to implement regression and classification models for machine learning algorithms. Then, evaluate each model and save and compare evaluation metrics. The data used in the regression analysis uses kaggle's 'red wine quilty', and the data used in the classification problem uses kaggle's 'Heart Failure Prediction'.

<hr>

## Data

![Static Badge](https://img.shields.io/badge/kaggle-Red%20Wine%20Quality-%2320BEFF?style=social&logoColor=white&labelColor=%2320BEFF)

![Static Badge](https://img.shields.io/badge/kaggle-Heart%20Failure%20Prediction-%2320BEFF?style=social&logoColor=white&labelColor=%2320BEFF)

- Regression

  - [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

- Classification

  - [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

<hr>

## Algorithm

- [D-Tree](#dtree)
- [RF](#rf)
- [NB](#nb)
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
- [K-NN](#knn)
- [Ada](#ada)
- [DA](#da)
  - Linear Discriminant Analysis
  - Quadratic Discriminant Analysis
- [SVM](#svm)
- [Voting](#voting)
- [Bagging](#bagging)

<hr>

<a name='dtree'></a>

### D-Tree **_(Decision Tree)_**

- **_Code_** [DTree.py](https://github.com/seunggihong/ML-Sklearn/blob/main/Algorithm/DTree.py)

- **_Hyper parameters_**
  ```json
  "dtree": { "max_depth": [1, 2, 3, 4, 5], "min_samples_split": [2, 3] }
  ```
- **_Usage_**
  ```bash
  $ python3 main.py --prob={reg or class} --model=dtree
  ```

<hr>
<a name='rf'></a>

### RF **_(Random Forest)_**

- **_Code_** [RF.py](https://github.com/seunggihong/ML-Sklearn/blob/main/Algorithm/RF.py)

- **_Hyper parameters_**
  ```json
  "rf": {
        "n_estimators": [10, 100],
        "max_depth": [6, 8, 10, 12],
        "min_samples_leaf": [8, 12, 18],
        "min_samples_split": [8, 16, 20]
      }
  ```
- **_Usage_**
  ```bash
  $ python3 main.py --prob={reg or class} --model=rf
  ```

<hr>

<a name='nb'></a>

### NB **_(Naive Bayes)_**

- **_Code_** [NB.py](https://github.com/seunggihong/ML-Sklearn/blob/main/Algorithm/NB.py)

**_Gaussian Naive Bayes(GNB)_**

- **_Hyper parameters_**
  ```json
  "gnb": {
        "var_smoothing": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
      }
  ```
- **_Usage_**
  ```bash
  $ python3 main.py --prob=class --model=gnb
  ```

**_Multinomial Naive Bayes(MNB)_**

- **_Hyper parameters_**
  ```json
  "mnb": {
        "var_smoothing": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
      }
  ```
- **_Usage_**
  ```bash
  $ python3 main.py --prob=class --model=mnb
  ```

<hr>

<a name='knn'></a>

### K-NN **_(K Nearest Neighbors)_**

- **_Code_** [KNN.py](https://github.com/seunggihong/ML-Sklearn/blob/main/Algorithm/KNN.py)

- **_Hyper parameters_**
  ```json
  "knn": {
        "n_neighbors": [1, 2, 3, 4, 5],
        "weights": ["uniform", "distance"]
      }
  ```
- **_Usage_**
  ```bash
  $ python3 main.py --prob={reg or class} --model=knn
  ```

<hr>

<a name='ada'></a>

### Ada **_(Adaptive Boosting)_**

- **_Code_** [Ada.py](https://github.com/seunggihong/ML-Sklearn/blob/main/Algorithm/Ada.py)

- **_Hyper parameters_**
  ```json
  "ada": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1]
      }
  ```
- **_Usage_**
  ```bash
  $ python3 main.py --prob={reg or class} --model=ada
  ```

<hr>

<a name='da'></a>

### DA **_(Discriminant Analysis)_**

- **_Code_** [DA.py](https://github.com/seunggihong/ML-Sklearn/blob/main/Algorithm/DA.py)

**_Linear Discriminant Analysis(LDA)_**

- **_Hyper parameters_**
  ```json
  "lda": {
        "n_components": [6, 8, 10, 12],
        "learning_decay": [0.75, 0.8, 0.85]
      }
  ```
- **_Usage_**
  ```bash
  $ python3 main.py --prob=class --model=lda
  ```

**_Quadratic Discriminant Analysis(QDA)_**

- **_Hyper parameters_**
  ```json
  Not yet
  ```
- **_Usage_**
  ```bash
  $ python3 main.py --prob=class --model=qda
  ```

<hr>

<a name='svm'></a>

### SVM **_(Support Vector Machine)_**

- **_Code_** [SVM.py](https://github.com/seunggihong/ML-Sklearn/blob/main/Algorithm/SVM.py)

- **_Hyper parameters_**
  ```json
  "svm": {
        "C": [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
        "kernel": ["linear", "rbf"],
        "gamma": [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
      }
  ```
- **_Usage_**
  ```bash
  $ python3 main.py --prob={reg or class} --model=svm
  ```

<hr>

<a name='voting'></a>

### Voting

- **_Code_** [Voting.py](https://github.com/seunggihong/ML-Sklearn/blob/main/Algorithm/Voting.py)

- **_Hyper parameters_**
  ```json
  Not yet
  ```
- **_Usage_**
  ```bash
  $ python3 main.py --prob={reg or class} --model=voting
  ```

<hr>

<a name='bagging'></a>

### Bagging

- **_Code_** [Bagging.py](https://github.com/seunggihong/ML-Sklearn/blob/main/Algorithm/Bagging.py)

- **_Hyper parameters_**
  ```json
  Not yet
  ```
- **_Usage_**
  ```bash
  $ python3 main.py --prob={reg or class} --model=bagging
  ```

<hr>

## Reference

- https://scikit-learn.org/stable/user_guide.html
