import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import hmmgmm
import lib
import parser
import warnings

warnings.filterwarnings("ignore")  # ignore Warnings

print('--------------------------- Data parsing ---------------------------------------------')
X_train, X_test, y_train, y_test, spk_train, spk_test = parser.parser('recordings', n_mfcc=13)
print('--------------------------- Data parsing Completed------------------------------------')

######################################### Step 9 ##############################################
print('\n--------------------------- Splitting Train Data----------------------------------------')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)
print('Data split 80% - 20% (stratified split).')
# Scale the data
scale_fn = parser.make_scale_fn(X_train)

X_train = scale_fn(X_train)
X_val = scale_fn(X_val)
X_test = scale_fn(X_test)

print('y train:')
print(np.asarray(np.unique(y_train, return_counts=True)).T)
print('\ny validation:')
print(np.asarray(np.unique(y_val, return_counts=True)).T)

######################################### Steps 10-12 ##############################################
# Getting a list which every index has the mfcc of that digit i.e. arranged_X[0] is for digit 0
arranged_X, seqlen = lib.arrange_digits(X_train, y_train)

print('\n--------------------------- Searching for Optimal parameters ----------------------------------------')
print('Parameters to be examined: n_componets = 1,2,3,4.\nn_mix=1,2,3,4,5.')
n_components = [1, 2, 3, 4]
n_mix = [1, 2, 3, 4, 5]
print('Metric: Accuracy.')
#  Return the optimal parameters
n_components, n_mix = hmmgmm.hmm_gridsearch(X_val, y_val, arranged_X, seqlen, n_components, n_mix, print_acc=True)

print('\n--------------------------- Modelling using Optimal parameters --------------------------------------')
# Creating 1 model for each digit using the train_gmmhmm we made using optimal parameters found by grid search
models = hmmgmm.train_gmmhmm(arranged_X, seqlen, n_components=n_components, n_mix=n_mix, covariance_type='diag',
                             n_iter=10,
                             verbose=False,
                             params='stmcw', init_params='mcw')

preds_test, accuracy = hmmgmm.predict_hmm(models, X_test, y_test)  # make predictions on the test data
preds_val, _ = hmmgmm.predict_hmm(models, X_val, y_val)  # make predictions on the test data
print(f'Accuracy score on Test data is {accuracy*100: .3f}%.')
print(f'Accuracy score on Validation data is {_*100: .3f}%.')
print(f'Accuracy score on both data is {(accuracy+_)*50: .3f}%.')

######################################### Steps 13 ##############################################
# Plotting confusion matrices for test and validation data
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))  # Set up the figure
ax = ax.ravel()

y = [y_val, y_test]
preds = [preds_val, preds_test]
titles = ['Validation Data', 'Test Data']
for i in range(2):
    actual = y[i]
    predicted =preds[i]  # get the prediction for each model (each loop)

    confusion_matrix = metrics.confusion_matrix(np.array(actual), np.array(predicted))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot(ax=ax[i])
    ax[i].set_title(titles[i])

fig.suptitle('GMM-HMM model using n_components=4 and n_mix=5')
plt.show()