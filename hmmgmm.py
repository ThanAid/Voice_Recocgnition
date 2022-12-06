import numpy as np
from hmmlearn.hmm import GMMHMM
import logging


def train_gmmhmm(X, seqlen, n_components=1, n_mix=1, covariance_type='diag', n_iter=10, verbose=False, params='stmcw',
                 init_params='stmcw'):
    """ Trains a gmm hmm model for each digit
    :arg X (List) containing ((array-like, shape (n_samples, n_features))
        Feature matrix of individual samples (for each digit).
    :arg seqlen (List) containing (array-like of integers, shape (n_sequences, ))
        – Lengths of the individual sequences in X. The sum of these should be n_samples for each digit
    :return models (List) containing all individual models (one for each digit)
        """

    startprob = np.concatenate((np.array([1.0]), np.zeros(n_components - 1)),
                               axis=0)  # initializing starting probabilities
    # transmats is a list with all the initialization for n_componets =1,2,3,4
    transmats = [np.array([[1.0]]), np.array([[0.5, 0.5], [0.0, 1.0]]),
                 np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]),
                 np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.0, 1.0]])]

    transmat = transmats[n_components - 1]  # initializing transmat

    models = []
    for i in range(len(X)):
        # Initializing each model
        model = GMMHMM(n_components=n_components, n_mix=n_mix, covariance_type=covariance_type, random_state=42,
                       n_iter=n_iter, verbose=verbose, params=params,
                       init_params=init_params)
        model.startprob_ = startprob
        model.transmat_ = transmat
        # fit
        model.fit(X[i], seqlen[i])
        models.append(model)

    return models


def predict_hmm(model, X, y):
    """ Calculates the predictions and the accuracy of given model
    :param model: List of hmm models
    :param X: array like containing np arrays (samples)
    :param y: array like containing the true values of X samples
    :return: preds (List) containing predictions
    :return: accuracy (float)
    """
    preds = []
    n_labels = len(np.unique(y))

    for x in X:
        prob = []  # Store the log probability under each model.
        for i in range(n_labels):
            prob.append(model[i].score(x))
        preds.append(prob.index(max(prob)))  # As a prediction we keep the index with the max probability

    accuracy = sum(np.array(preds) == np.array(y)) / len(y)  # Calculate the accuracy by comparing preds with y

    return preds, accuracy


def hmm_gridsearch(X, y, arranged_X, seqlen, n_components, n_mix, print_acc=True):
    """ Searches for optimal parameters
    :param seqlen: (List) containing (array-like of integers, shape (n_sequences, ))
        – Lengths of the individual sequences in X. The sum of these should be n_samples for each digit (train)
    :param arranged_X: X (List) containing ((array-like, shape (n_samples, n_features))
        Feature matrix of individual samples (for each digit). (train)
    :param y: array like containing the true values of X samples (validation)
    :param X: array like containing np arrays (samples) (validation)
    :param model: List of hmm models
    :param n_components: List containing values we want to check
    :param n_mix: List containing values we want to check
    :return: optimal parameters
    """
    logging.getLogger("hmmlearn").setLevel("CRITICAL")
    scores = []  # List to store accuracies to find optimal
    __ = []  # List to keep track of n_comp and n_mix
    for comp in n_components:
        for mix in n_mix:
            model = train_gmmhmm(arranged_X, seqlen, n_components=comp, n_mix=mix, covariance_type='diag', n_iter=10,
                                  verbose=False,
                                  params='stmcw', init_params='mcw')
            _, accuracy = predict_hmm(model, X, y)
            __.append([comp, mix])
            scores.append(accuracy)
            if print_acc:
                print(f'Accuracy for n_components={comp} and n_mix={mix} is {accuracy}')

    best = __[scores.index(max(scores))]
    print('-----------------------------------------------------------------------------------------------')
    print(f'The best parameters with {max(scores)} accuracy are: \nn_components={best[0]}\nn_mix={best[1]}')

    return best[0], best[1]
