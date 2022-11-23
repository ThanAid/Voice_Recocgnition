import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
import time

def optimise_GaussianNB(X_train, y_train, X_test, y_test):
    # Setting the values that will be tested via Gridsearch
    vthreshold = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]  # Variance Threshold (We don't want to lose too many features)
    var_smooth = np.logspace(-2, -11, num=20)  # Gaussian NB hyperparameter
    selector = VarianceThreshold()  # Initializing VarianceThreshold
    clf = GaussianNB()

    # Initializing the pipeline and the gridsearch for accuracy testing
    pipe = Pipeline(steps=[('selector', selector), ('GaussianNB', clf)])
    estimator = GridSearchCV(pipe, dict(selector__threshold=vthreshold,
                                        GaussianNB__var_smoothing=var_smooth), cv=10, scoring='accuracy', n_jobs=-1)

    print('---------------Optimizing via Accuracy scoring------------------')

    start_time = time.time()
    Gaussian_NB_opt = estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    stop_time = time.time()
    print("Total fit and predict time: %s seconds" % (stop_time - start_time))
    print(classification_report(y_test, preds))

    print('Optimized Gaussian NB model via accuracy is:')
    print(estimator.best_estimator_)
    print(estimator.best_params_)


def optimise_kNN(X_train, y_train, X_test, y_test):
    # Setting the values that will be tested via Gridsearch
    vthreshold = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]  # Variance Threshold (We don't want to lose too many features)
    n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # kNN hyperparameter
    selector = VarianceThreshold()  # Initializing VarianceThreshold
    clf = KNeighborsClassifier()

    # Initializing the pipeline and the gridsearch for accuracy testing
    pipe = Pipeline(steps=[('selector', selector), ('kNN', clf)], memory='tmp')
    estimator = GridSearchCV(pipe, dict(selector__threshold=vthreshold,
                                        kNN__n_neighbors=n_neighbors), cv=10, scoring='accuracy', n_jobs=-1)

    print('---------------Optimizing via Accuracy scoring------------------')
    start_time = time.time()
    kNN_opt = estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    stop_time = time.time()
    print("Total fit and predict time: %s seconds" % (stop_time - start_time))
    print(classification_report(y_test, preds))

    print('Optimized kNN model via accuracy is:')
    print(estimator.best_estimator_)
    print(estimator.best_params_)


def optimise_SVC(X_train, y_train, X_test, y_test):
    # SVM hyperparameters
    kernel = ['poly', 'rbf', 'sigmoid']
    gamma = ['auto', 'scale']
    C = [0.1, 1, 1.5]
    clf = SVC()

    # Initializing the pipeline and the gridsearch for accuracy testing
    pipe = Pipeline(steps=[('SVC', clf)], memory='tmp')
    estimator = GridSearchCV(pipe, dict(SVC__C=C, SVC__kernel=kernel, SVC__gamma=gamma), cv=10, scoring='accuracy',
                             n_jobs=-1)
    # Initializing the gridsearch for f1 weigthed testing
    estimator_f1 = GridSearchCV(pipe, dict(SVC__C=C, SVC__kernel=kernel, SVC__gamma=gamma), cv=10,
                                scoring='f1_weighted', n_jobs=-1)

    print('---------------Optimizing via Accuracy scoring------------------')
    start_time = time.time()
    SVC_opt = estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    stop_time = time.time()
    print("Total fit and predict time: %s seconds" % (stop_time - start_time))
    print(classification_report(y_test, preds))

    print('Optimized LR model via accuracy is:')
    print(estimator.best_estimator_)
    print(estimator.best_params_)


def optimise_LR(X_train, y_train, X_test, y_test):
    # Setting the values that will be tested via Gridsearch
    vthreshold = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]  # Variance Threshold (We don't want to lose too many features)
    # LR hyperparameters
    C = [0.1, 0.5, 1, 1.5]
    penalty = ['l2']

    selector = VarianceThreshold()  # Initializing VarianceThreshold
    clf = LogisticRegression()

    # Initializing the pipeline and the gridsearch for accuracy testing
    pipe = Pipeline(steps=[('selector', selector), ('LR', clf)], memory='tmp')
    estimator = GridSearchCV(pipe,
                             dict(selector__threshold=vthreshold, LR__solver=['sag'],
                                  LR__penalty=penalty, LR__C=C), cv=10,
                             scoring='accuracy', n_jobs=-1)

    print('---------------Optimizing via Accuracy scoring------------------')
    start_time = time.time()
    LR_opt = estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    stop_time = time.time()
    print("Total fit and predict time: %s seconds" % (stop_time - start_time))
    print(classification_report(y_test, preds))

    print('Optimized LR model via accuracy is:')
    print(estimator.best_estimator_)
    print(estimator.best_params_)