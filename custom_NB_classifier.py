import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def digit_mean(X, y, digit):
    """Calculates the mean for all instances of a specific digit
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    """
    X_digit = X[y == digit]
    mean_array = np.mean(X_digit, axis=0)

    return mean_array


def digit_variance(X, y, digit):
    """Calculates the variance for all instances of a specific digit
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    """
    X_digit = X[y == digit]
    var_array = np.var(X_digit, axis=0)

    return var_array


def calculate_priors(y):
    """Return the a-priori probabilities for every class
    Args:
        y (np.ndarray): Labels for dataset (nsamples)
    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    counted = []
    n = len(y)
    # Number of unique values on y
    digits = np.sort(np.unique(y))

    for digit in digits:
        counted.append(np.count_nonzero(y == digit))

    return np.array([a / n for a in counted])


class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier
     var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability."""

    def __init__(self, use_unit_variance=False, var_smoothing=1e-9):
        self.use_unit_variance = use_unit_variance
        self.X_mean_ = None
        self.X_var_ = None
        self.priors_ = None
        self.var_smoothing = var_smoothing
        self.labels = None

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        Calculates self.X_mean_ based on the mean
        feature values in X for each class.
        Calculates self.X_var_ based on the variance
        feature values in X for each class.
        self.X_mean_  and self.X_var_ becomes a numpy.ndarray of shape
        (n_classes, n_features)
        fit always returns self.
        """
        self.labels = np.sort(np.unique(y))
        mean_arr = np.empty([len(self.labels), X.shape[1]])
        var_arr = np.empty_like(mean_arr)

        k = 0
        for i in self.labels:
            mean_arr[k] = digit_mean(X, y, i)  # Calculate the mean values of features
            var_arr[k] = digit_variance(X, y, i)  # Calculate the variance values of features
            k += 1

        counted = calculate_priors(y)

        self.X_mean_ = mean_arr
        if not self.use_unit_variance:
            self.X_var_ = var_arr
        else:
            self.X_var_ = np.ones_like(
                var_arr)  # if use_unit_variance is True then the variance values will be set as 1
        self.priors_ = counted
        print()
        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        Bayes Theorem
        Args:
            X (np.ndarray): Digits data (nsamples x nfeatures)
        """
        preds = []
        # adding a small value to the variance values of the features to avoid being treated as 0 if they are too small
        var = self.X_var_ + (self.var_smoothing * self.X_var_.max())

        for i in range(len(X)):  # Calculating how many digits we have stored in X
            posterior = []  # list that will store posterior

            for digit in range(len(self.labels)):
                pxc = np.prod((1 / (np.sqrt((2 * np.pi * var[digit])))) * np.exp(  # Calculating the Π(P(x|y))
                    -np.power(X[i] - self.X_mean_[digit], 2) / (2 * var[digit])))

                posterior.append(np.log(pxc) + np.log(self.priors_[digit]))  # appending the result from discriminant
                # function for that digit (possibility for that sample to be classified as that digit)

            pred_ind = posterior.index(max(posterior))
            preds.append(np.sort(self.labels)[pred_ind])
            # append the index of the max value of the posterior list,
            # meaning that the sample gets classified as the digit with the max posterior

        return np.array(preds)

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """

        return sum(self.predict(X) == y) / len(y)
