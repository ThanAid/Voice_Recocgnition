from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from custom_NB_classifier import CustomNBClassifier
import lib
import optimising_clfs
from imblearn.pipeline import Pipeline
import time
import warnings

warnings.filterwarnings("ignore")  # ignore Warnings

######################### Step 2 #####################################
wavs, y, speakers, fnames = lib.parse_free_digits('digits')
print('\nData parsing completed.')
print('---------------------------------------------------------------')

######################### Step 3 #####################################
mfccs, delta1, delta2 = lib.extract_features(wavs, window=25, step=10, n_mfcc=13, Fs=16000)
print('---------------------------------------------------------------')

######################### Step 4 #####################################
print('\nPlotting Histogram.')
# Plot Histogram
lib.plot_hist(mfccs, ['one', 'nine'], y)
print('---------------------------------------------------------------')
print('\nGetting MFSCs.')
# Get MFSC for all the digits
mfscs = lib.extract_mfscs(wavs, window=25, n_mels=13, step=10, Fs=16000, norm=None)
# Get MFSCS for specified digits and number of speakers
mfscs_1_1, mfscs_1_2, mfscs_2_1, mfscs_2_2 = lib.corr_matrices(y=mfscs, digits=['one', 'nine'], n_speakers=2,
                                                               name_list=y, corr=True)
print('---------------------------------------------------------------')
print('\nPlotting Correlation matrices.')
# Plot correlation matrices for them
lib.plot_correlation_matrix(mfscs_1_1, mfscs_1_2, mfscs_2_1, mfscs_2_2, digits=['one', 'nine'], method='MFSC')
# Get MFCCS for specified digits and number of speakers
mfccs_1_1, mfccs_1_2, mfccs_2_1, mfccs_2_2 = lib.corr_matrices(y=mfccs, digits=['one', 'nine'], n_speakers=2,
                                                               name_list=y, corr=True)
# Plot correlation matrices for them
lib.plot_correlation_matrix(mfccs_1_1, mfccs_1_2, mfccs_2_1, mfccs_2_2, digits=['one', 'nine'], method='MFCC')
print('---------------------------------------------------------------')

######################### Step 5 #####################################
print('\nCreating DataFrame.\n')
# Get means and vars for all the features and put them in a df
df = lib.features_to_df(mfccs, delta1, delta2, y, speakers)
print(df.head())
print('---------------------------------------------------------------')
print('\nPlotting Scatter Plots.')
# Scatter for 1st MFCC
lib.plot_scatter(df, 'mean_mfcc_1', 'var_mfcc_1', xlabel='Mean for 1st MFCC', ylabel='Variance for 1st MFCC',
                 labels='class', method='MFCC')
# Scatter for 1st delta
lib.plot_scatter(df, 'mean_delta1_1', 'var_delta1_1', xlabel='Mean for 1st delta', ylabel='Variance for 1st delta',
                 labels='class', method='delta')
# Scatter for 1st delta-deltas
lib.plot_scatter(df, 'mean_delta2_1', 'var_delta2_1', xlabel='Mean for 1st delta-deltas',
                 ylabel='Variance for 1st delta-deltas', labels='class', method='delta-deltas')
print('---------------------------------------------------------------')

######################### Step 6 #####################################
# in a X DataFrame we only keep the features and in y the labels
X = df.drop(['speaker', 'class'], axis=1)
y = df['class']

# Standardization (recommended for PCA)
X_s = StandardScaler().fit_transform(X)

# Using PCA and keeping 2 components
pca2 = PCA(n_components=2)
X_2 = pca2.fit_transform(X_s)
# Creating a new DataFrame to use on the scatter plots using the features from pca(n=2)
df_2 = pd.DataFrame()
df_2['V1'] = X_2[:, 0]
df_2['V2'] = X_2[:, 1]
df_2['class'] = y

# Scatter for PCA n=2
lib.plot_scatter(df_2, 'V1', 'V2', xlabel='Principal Component 1', ylabel='Principal Component 2', labels='class',
                 method='PCA', title='Using PCA keeping 2 components')

# Using PCA and keeping 3 components
pca3 = PCA(n_components=3)
X_3 = pca3.fit_transform(X_s)

# Creating a new DataFrame to use on the scatter plots using the features from pca(n=2)
df_3 = pd.DataFrame()
df_3['V1'] = X_3[:, 0]
df_3['V2'] = X_3[:, 1]
df_3['V3'] = X_3[:, 2]
df_3['class'] = y

# Scatter for PCA n=3
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(df_3['V1'], df_3['V2'], df_3['V3'], c=df_3['class'])
# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="best", title="Classes")
ax.add_artist(legend1)
ax.set_title('Using PCA keeping 3 components')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()

# Variance of PCA(n=2) and PCA(n=3) components
print(f'The variance of the data without using PCA with 2 components is : {np.var(X_2)}. '
      f'That is {pca2.explained_variance_ratio_.cumsum()[-1] * 100} % of starting variance.')
print(f'The variance of the data without using PCA with 3 components is : {np.var(X_3)}. '
      f'That is {pca3.explained_variance_ratio_.cumsum()[-1] * 100} % of starting variance.')

######################### Step 7 #####################################
# Data Normalization
X = StandardScaler().fit_transform(X)
# split the data to train data and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('---------------------------------------------------------------------------------------------------------------')
print('----------------------------------------OUT OF THE BOX CLASSIFIERS---------------------------------------------')
print('---------------------------------------------------------------------------------------------------------------')
# Using the model created on the last lab exercise (Custom Naive Bayes)
print('\nModelling using Custom Naive Bayes model:')
model = CustomNBClassifier()
model.fit(X_train, y_train)
print(f'Score for the custom Naive Bayes classifier on test data: {model.score(X_test, y_test.to_numpy()) * 100} %.')
cross_score, score_std = lib.evaluate_classifier(model, X_train, y_train, folds=10)
print(f'Score estimated via cross-validation with 5 folds is: {cross_score * 100} \u00B1 {score_std * 100}%.')
print('---------------------------------------------------------------------------------------------------------------')

print('\nModelling using Gaussian Naive Bayes model:')
# Using the model Gaussian Naive Bayes from sklearn
model = GaussianNB()
model.fit(X_train, y_train)
print(f'Score for the Gaussian Naive Bayes classifier on test data: {model.score(X_test, y_test) * 100} %.')
cross_score, score_std = lib.evaluate_classifier(model, X_train, y_train, folds=10)
print(f'Score estimated via cross-validation with 5 folds is: {cross_score * 100} \u00B1 {score_std * 100}%.')
print('---------------------------------------------------------------------------------------------------------------')

print('\nModelling using kNN model:')
# Using the model kNN from sklearn
model = KNeighborsClassifier()
model.fit(X_train, y_train)
print(f'Score for the kNN classifier on test data: {model.score(X_test, y_test) * 100} %.')
cross_score, score_std = lib.evaluate_classifier(model, X_train, y_train, folds=10)
print(f'Score estimated via cross-validation with 5 folds is: {cross_score * 100} \u00B1 {score_std * 100}%.')
print('---------------------------------------------------------------------------------------------------------------')

print('\nModelling using SVC model:')
# Using the model SVC from sklearn
model = SVC()
model.fit(X_train, y_train)
print(f'Score for the SVC classifier on test data: {model.score(X_test, y_test) * 100} %.')
cross_score, score_std = lib.evaluate_classifier(model, X_train, y_train, folds=10)
print(f'Score estimated via cross-validation with 5 folds is: {cross_score * 100} \u00B1 {score_std * 100}%.')
print('---------------------------------------------------------------------------------------------------------------')

print('\nModelling using Logistic Regression model:')
# Using the model Logistic Regression from sklearn
model = LogisticRegression()
model.fit(X_train, y_train)
print(f'Score for the Logistic Regression classifier on test data: {model.score(X_test, y_test) * 100} %.')
cross_score, score_std = lib.evaluate_classifier(model, X_train, y_train, folds=10)
print(f'Score estimated via cross-validation with 5 folds is: {cross_score * 100} \u00B1 {score_std * 100}%.')
print('---------------------------------------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------------------------------------')

print(
    '\n--------------------------------------------------------------------------------------------------------------')
print(
    '-------------------------------------------OPTIMISING CLASSIFIERS-----------------------------------------------')
print(
    '----------------------------------------------------------------------------------------------------------------')
# Creating a dictionary to store optimized models (scores and predictions)
mod_opt = {}

# optimising_clfs.optimise_GaussianNB(X_train, y_train, X_test, y_test) # UNCOMMENT TO GRIDSEARCH FOR OPT GAUSSIAN NB
print('\nModelling using Gaussian Naive Bayes model:')
# Create the pipeline for training the optimal Gaussian model
pipe_gnb = Pipeline(steps=[('selector', VarianceThreshold(threshold=0.8)), ('pca', PCA()),
                           ('GaussianNB', GaussianNB(var_smoothing=0.01))])
# Fit, test and store the scores via the function we made before
mod_opt = lib.make_opt_mod(pipe_gnb, X_train, y_train, X_test, y_test, 'GaussianNB', mod_dict=mod_opt)
print(
    '\n--------------------------------------------------------------------------------------------------------------')

# optimising_clfs.optimise_kNN(X_train, y_train, X_test, y_test) # UNCOMMENT TO GRIDSEARCH FOR OPT kNN
print('\nModelling using kNN model:')
pipe_kNN = Pipeline(memory='tmp',
                    steps=[('selector', VarianceThreshold(threshold=0)),
                           ('kNN', KNeighborsClassifier(n_neighbors=1))])
# Fit, test and store the scores via the function we made before
mod_opt = lib.make_opt_mod(pipe_kNN, X_train, y_train, X_test, y_test, 'kNN', mod_dict=mod_opt)
print(
    '\n--------------------------------------------------------------------------------------------------------------')

# optimising_clfs.optimise_SVC(X_train, y_train, X_test, y_test) # UNCOMMENT TO GRIDSEARCH FOR OPT kNN
print('\nModelling using SVC model:')
pipe_SVC = Pipeline(memory='tmp', steps=[('SVC', SVC(C=1.5, gamma='auto'))])
# Fit, test and store the scores via the function we made before
mod_opt = lib.make_opt_mod(pipe_SVC, X_train, y_train, X_test, y_test, 'SVC', mod_dict=mod_opt)
print(
    '\n--------------------------------------------------------------------------------------------------------------')

# optimising_clfs.optimise_LR(X_train, y_train, X_test, y_test) # UNCOMMENT TO GRIDSEARCH FOR OPT kNN
print('\nModelling using LR model:')
pipe_LR = Pipeline(memory='tmp',
         steps=[('selector', VarianceThreshold(threshold=0.8)),
                ('LR', LogisticRegression(C=1, solver='sag'))])
# Fit, test and store the scores via the function we made before
mod_opt = lib.make_opt_mod(pipe_LR, X_train, y_train, X_test, y_test, 'LR', mod_dict=mod_opt)
print('---------------------------------------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------------------------------------')