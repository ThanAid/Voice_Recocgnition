import os
from datetime import datetime
from glob import glob
import librosa
import re
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import torch
import torch.nn as nn
from statistics import mean
from pytorchtools import EarlyStopping


def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("\\")[1].split(".")[0].split("_")[0] for f in files]  # get all file names
    y = [re.split(r'(\d+)', f)[0] for f in fnames]  # Get the Digit names (for example 'eight')
    speakers = [int(re.split(r'(\d+)', f)[1]) for f in fnames]  # Get the speaker names (for example 1)

    _, Fs = librosa.core.load(files[0], sr=None)  # Get the sample rate
    print(f'The sample rate is: {Fs} Hz.')

    def read_wav(f):
        wav, sr = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}".format(len(wavs)))

    return wavs, y, speakers, fnames


def extract_features(y, window=25, step=10, n_mfcc=13, Fs=16000, norm=None):
    """Calculates the MFCCS , delta and delta-deltas values
       Args:
           y (np.ndarray): sound sample (nsamples)
           window (int): window length (in ms)
           step (int): step (in ms)
           n_mfcc (int): number of NFCCS you want to extract
           Fs (int): Sample freq in Hz
       Returns:
           (list) mfccs
           (list) delta
           (list) deltas-delta
       """
    mfccs = []  # list to store 13 mfccs values for each file (mfccs values (ndarray))
    delta1 = []  # list to store delta values for each file (delta values (ndarray))
    delta2 = []  # list to store delta-deltas values for each file ( deltas values (ndarray))
    window = int(window * 16000 / 1000)
    step = int(step * 16000 / 1000)

    mfccs = [librosa.feature.mfcc(y=y, sr=16000, n_mfcc=n_mfcc, n_fft=window, hop_length=step).T for y in
             tqdm(y, desc="Extracting mfcc features...")]
    for i in range(len(y)):
        delta1.append(librosa.feature.delta(data=mfccs[i], order=1))
        delta2.append(librosa.feature.delta(data=mfccs[i], order=2))

    print('\nFeature extraction completed for all the data.')

    return mfccs, delta1, delta2


def find_indices(list_to_check, item_to_find):
    # Find the indices of a specific value on a list
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices


def plot_hist(mfcc, digits, name_list):
    """Plots histogram of given digits
       Args:
           mfcc (list)
           digits (list): list containing digits (str) you want to plot (if you want more than 2 digits it needs changes)
           name_list (list): List containing the names of the files.
       """

    first_digit_indeces = find_indices(name_list, digits[0])  # Get indeces for first digit
    second_digit_indeces = find_indices(name_list, digits[1])  # Get indeces for second digit

    first_mfccs = [mfcc[i] for i in first_digit_indeces]  # Get mfccs for all the values for first digit
    second_mfccs = [mfcc[i] for i in second_digit_indeces]  # Get mfccs for all the values for second digit

    first_mfccs_ = first_mfccs[0]  # Define a starting value for the loop below
    second_mfccs_ = second_mfccs[0]  # Define a starting value for the loop below

    for i in range(len(first_digit_indeces) - 1):
        # Combining all the MFCCS for each digit (For every sample of that digit)
        first_mfccs_ = np.concatenate((first_mfccs_, first_mfccs[i + 1]), axis=0)  # For the 1st digit
        second_mfccs_ = np.concatenate((second_mfccs_, second_mfccs[i + 1]), axis=0)  # For the 2nd digit

    # Defining the bins for the Histograms
    bins = np.linspace(min(min(first_mfccs_[:, 0]), min(second_mfccs_[:, 0])),
                       max(max(first_mfccs_[:, 0]), max(second_mfccs_[:, 0])), 20)
    bins2 = np.linspace(min(min(first_mfccs_[:, 1]), min(second_mfccs_[:, 1])),
                        max(max(first_mfccs_[:, 1]), max(second_mfccs_[:, 1])), 20)

    # Plotting the histograms
    fig = plt.figure(figsize=(15, 5))
    # 1st MFCC
    plt.subplot(1, 2, 1)
    plt.hist(first_mfccs_[:, 0], bins, alpha=0.7, color='navy', edgecolor='black', linewidth=1.2,
             label=f'Digit: {digits[0]}')
    plt.hist(second_mfccs_[:, 0], bins, alpha=0.7, color='darkgray', edgecolor='black', linewidth=1.2,
             label=f'Digit: {digits[1]}')
    plt.title('1st MFCC')
    plt.legend(loc=4)

    # 2nd MFCC
    plt.subplot(1, 2, 2)
    plt.hist(first_mfccs_[:, 1], bins2, alpha=0.7, color='navy', edgecolor='black', linewidth=1.2,
             label=f'Digit: {digits[0]}')
    plt.hist(second_mfccs_[:, 1], bins2, alpha=0.7, color='darkgray', edgecolor='black', linewidth=1.2,
             label=f'Digit: {digits[1]}')
    plt.title('2nd MFCC')
    plt.legend(loc=4)

    plt.legend(loc=4)
    plt.suptitle(f'MFCC Coefficient for digits {digits[0]} and {digits[1]}')
    plt.show()


def extract_mfscs(y, window=25, step=10, n_mels=13, Fs=16000, norm=None):
    """Calculates the MFCCS , delta and delta-deltas values
           Args:
               y (np.ndarray): sound sample (nsamples)
               window (int): window length (in ms)
               step (int): step (in ms)
               n_mels (int): number of NFSCS you want to extract
               Fs (int): Sample freq in Hz
               norm {None, ‘slaney’, or number} [scalar]
           Returns:
               (list) mfscs
           """
    window = window * 16000 // 1000
    step = step * 16000 // 1000
    mfscs = []  # list to store 13 mfscs values for each file (mfscs values (ndarray))

    mfscs = [librosa.feature.melspectrogram(y=y, sr=16000, n_mels=n_mels, n_fft=window, hop_length=step,
                                            norm=norm).T for y in tqdm(y, desc="Extracting mfsc features...")]

    return mfscs


def corr_matrices(y, digits, n_speakers, name_list, corr=True):
    """Calculates the MFSCS/MFCCS and the correlation matrices for specified digit/speaker
               Args:
                   y (np.ndarray): MFSCS
                   digits (list): list containing strings for ex. 'nine'
                   n_speakers (int): number of speakers u want to keep
                   name_list (list): list containing names of files for ex. 'eight1'
                   corr (Boolean): True if you want the correlation matrices
               Returns:
                   mfscs_1_1 (np.ndarray): containing mfscs matrix for the first digit and first speaker
                   mfscs_1_2 (np.ndarray): containing mfscs matrix for the first digit and second speaker
                   mfscs_2_1 (np.ndarray): containing mfscs matrix for the second digit and first speaker
                   mfscs_2_2 (np.ndarray): containing mfscs matrix for the second digit and second speaker
               """
    first_digit_indeces = find_indices(name_list, digits[0])  # Get indeces for first digit
    second_digit_indeces = find_indices(name_list, digits[1])

    # Get MFSCS for the given digits and speakers
    mfscs_1_1 = y[first_digit_indeces[0]]
    mfscs_1_2 = y[first_digit_indeces[1]]
    mfscs_2_1 = y[second_digit_indeces[0]]
    mfscs_2_2 = y[second_digit_indeces[1]]

    if corr:
        # if corr is True then get the correlation matrices
        mfscs_1_1 = np.corrcoef(mfscs_1_1.T)
        mfscs_1_2 = np.corrcoef(mfscs_1_2.T)
        mfscs_2_1 = np.corrcoef(mfscs_2_1.T)
        mfscs_2_2 = np.corrcoef(mfscs_2_2.T)

    return mfscs_1_1, mfscs_1_2, mfscs_2_1, mfscs_2_2


def plot_correlation_matrix(y1, y2, y3, y4, digits, method):
    # Plotting the correlation matrices
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    # 1st MFCC
    a1 = ax1.matshow(y1)
    fig.colorbar(a1)
    ax1.set_title(f'1st {method} of Digit {digits[0]}')

    # 2nd MFCC
    a2 = ax2.matshow(y2)
    fig.colorbar(a2)
    ax2.set_title(f'2nd {method} of Digit {digits[0]}')

    # 1st MFCC
    a3 = ax3.matshow(y3)
    fig.colorbar(a3)
    ax3.set_title(f'1st {method} of Digit {digits[1]}')

    # 2nd MFCC
    a4 = ax4.matshow(y4)
    fig.colorbar(a4)
    ax4.set_title(f'1st {method} of Digit {digits[0]}')

    plt.show()


def convert_str_int(labels):
    '''Converts labels from strings to ints
        :arg labels (list): contains labels in strings
        '''
    # first we will find the indeces for each digit
    lab_dict = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }
    for key in lab_dict.keys():
        indeces = find_indices(labels, key)
        for index in indeces:
            labels[index] = lab_dict[key]

    return labels


def features_to_df(feature1, feature2, feature3, labels, speakers):
    '''Calculates mean and variance for each feature given and creates a df with all the features
        :param
            feature1 (np.ndarray)
            feature2 (np.ndarray)
            feature3 (np.ndarray)
            labels (list)
            speakers (list)
        :return
            df (pd.DataFrame)
        '''

    feats = []  # Store all the mean and var of all features
    titles = []  # Store columnn names for the df
    columns = len(feature1[0][0, :])
    # For MFCC
    for i in range(columns):
        titles.append(f'mean_mfcc_{i + 1}')
        titles.append(f'var_mfcc_{i + 1}')
    # for delta
    for i in range(columns):
        titles.append(f'mean_delta1_{i + 1}')
        titles.append(f'var_delta1_{i + 1}')
    # for delta-deltas
    for i in range(columns):
        titles.append(f'mean_delta2_{i + 1}')
        titles.append(f'var_delta2_{i + 1}')

    df = pd.DataFrame()  # Create empty dataframe to store the means and vars
    k = 0
    for j in range(columns):  # for mfccs
        df[titles[k]] = [np.mean(feature1[i][:, j]) for i in range(len(feature1))]
        df[titles[k + 1]] = [np.var(feature1[i][:, j]) for i in range(len(feature1))]
        k += 2

    for j in range(columns):  # for delta
        df[titles[k]] = [np.mean(feature2[i][:, j]) for i in range(len(feature2))]
        df[titles[k + 1]] = [np.var(feature2[i][:, j]) for i in range(len(feature2))]
        k += 2

    for j in range(columns):  # for delta-deltas
        df[titles[k]] = [np.mean(feature3[i][:, j]) for i in range(len(feature3))]
        df[titles[k + 1]] = [np.var(feature3[i][:, j]) for i in range(len(feature3))]
        k += 2

    df['speaker'] = speakers
    # for the class column to avoid having string values we will now convert every string to the corresponding digit
    # if all(isinstance(labels, int) for item in labels): #Uncomment if you want to first check if labels are ints already
    #     df['class'] = labels

    labels = convert_str_int(labels)  # Comment if labels are already ints
    df['class'] = labels

    return df


def plot_scatter(df, feat1, feat2, labels, xlabel=None, ylabel=None, method=None, title=None):
    """Plots scatter plot for 2 specified features
            Args:
                df (pd.DataFrame)
                feat1 (str): name of feature 1
                feat2 (str): name of feature 2
                labels (str): name of column with the labels
                xlabel (str)
                ylabel (str)
                method (str)
               """
    fig, ax = plt.subplots()

    # scatter = ax.scatter(feat1, feat2, c=labels, marker=markers)
    scatter = sns.scatterplot(data=df, x=feat1, y=feat2, hue=labels, style=labels, legend='full')
    # add title, xlabel and ylabel
    if title == None:
        scatter.set(title=f'Scatterplot of the mean and Variance of 1st {method}')
    else:
        scatter.set(title=title)
    scatter.set(xlabel=xlabel)
    scatter.set(ylabel=ylabel)

    plt.show()


def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y
    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        folds (int): the number of folds
    Returns:
        (float): The 5-fold classification mean score and std(accuracy)
    """
    results = cross_val_score(estimator=clf, X=X, y=y, cv=folds)
    accur = results.mean()
    accur_std = results.std()
    return accur, accur_std


def make_opt_mod(pipe, X_train, y_train, X_test, y_test, mod_name, mod_dict={}):
    '''Fits, evaluates and stores the model and its scores to a dictionary and prints the scores
      Args:
            pipe: pipeline for the optimal model
            mod_name(str): Model name
            mod_dict(dictionary): a dictionary to store the model
      Returns:
            mod_dict(dictionary): updated dictionary
            '''
    start_time = time.time()  # train counter
    mod = pipe.fit(X_train, y_train)
    stop_time = time.time()
    print(f'Fit time for {mod_name} model is: {stop_time - start_time: .2f} seconds.')
    start_time = time.time()  # test counter
    preds = mod.predict(X_test)
    stop_time = time.time()
    print(f'Predict time for {mod_name} model is: {stop_time - start_time: .2f} seconds.')
    # Store to the opt mod dictionary
    mod_dict[mod_name] = [mod.score(X_test, y_test), f1_score(y_test, mod.predict(X_test), average='weighted'),
                          cross_val_score(mod, X_train, y_train, cv=10), preds]
    print(
        f'{mod_name} has {mod_dict[mod_name][0] * 100: .3f}% accuracy and {mod_dict[mod_name][1] * 100: .3f}% F1 score and {mod_dict[mod_name][2].mean() * 100: .3f}% +- {mod_dict[mod_name][2].std() * 100: .3f}% 10 fold-cv.')

    return mod_dict


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Number of hidden layers
        self.n_layers = n_layers
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # RNN
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)  # batch_first means x needs to be: (
        # batch_size, seq, input_size)
        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # out (batch_size, time_step, hidden_size)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)

        # get RNN outputs
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


class RNN_GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN_GRU, self).__init__()

        # Number of hidden layers
        self.n_layers = n_layers
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # RNN
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)  # batch_first means x needs to be: (
        # batch_size, seq, input_size)
        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # out (batch_size, time_step, hidden_size)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)

        # get RNN outputs
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN_LSTM, self).__init__()

        # Number of hidden layers
        self.n_layers = n_layers
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # RNN
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)  # batch_first means x needs to be: (
        # batch_size, seq, input_size)
        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # out (batch_size, time_step, hidden_size)

        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
        # Initial cell state (zeroes)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()

        # get RNN outputs
        out, (_, __) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)

        return out


def train_model(model, train_loader, optimizer, criterion, n_epochs=100, batch_size=32, n_features=1, val=None):
    # Train the model
    n_total_steps = len(train_loader)
    for epoch in range(n_epochs):
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.view([batch_size, -1, n_features])
            # Clear gradients
            optimizer.zero_grad()

            prediction = model(x_batch)
            # Forward pass

            loss = criterion(prediction, y_batch)
            # Backward and optimize

            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

        if val is not None:
            model.eval()  # turns off batchnorm/dropout ...
            acc = 0
            n_samples = 0
            with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
                for i, data in enumerate(val):
                    X_batch, y_batch = data  # test data and labels
                    X_batch = X_batch.view([batch_size, -1, n_features])

                    preds = model(X_batch)  # get net's predictions
                    loss = criterion(preds, y_batch)

            if epoch % 1 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Loss (on validation): {loss.item():.4f}')

    return model


def predict_model(model, test_loader, batch_size, n_features, criteriion):
    # Make predictions using models
    preds = []
    true_values = []
    accu = 0
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.view([batch_size, -1, n_features])

        pred = (model(x_batch))
        preds.append(pred.detach().numpy())  # predict
        true_values.append(y_batch.detach().numpy())
        accu += criteriion(pred, y_batch)
    return preds, true_values, (1 - accu)


def arrange_digits(X, y):
    # splits data based on the unique labels on y
    n_labels = np.unique(y)  # Find all labels

    arranged = []
    X_final = []
    seqlen = []

    for i in n_labels:
        _ = []
        index = find_indices(y, i)  # find indexes for that digit
        arranged.append([X[j] for j in index])  # keep values of those indexes

        X_arranged = arranged[i][0]  # Initialization of the first sample for that digit
        _.append(len(X_arranged))
        for j in range(1, len(arranged[i])):
            X_arranged = np.concatenate((X_arranged, arranged[i][j]))  # concatenate all samples for that digit
            _.append(len(arranged[i][j]))

        seqlen.append(_)
        X_final.append(X_arranged)

    return X_final, seqlen


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout=0, bidirectional=False):
        super(LSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = hidden_dim * 2 if self.bidirectional else hidden_dim
        # Dropout probability
        self.dropout = dropout
        # Number of hidden layers
        self.n_layers = n_layers
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # RNN
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)  # batch_first means x needs to be: (
        # batch_size, seq, input_size)
        # last, fully-connected layer
        self.fc = nn.Linear(self.feature_size, output_size)

    def forward(self, x):
        d_layers = self.n_layers * 2 if self.bidirectional else self.n_layers
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # out (batch_size, time_step, hidden_size)
        # Unpacking x (packed in train)
        seq_unpacked, lens_unpacked = pad_packed_sequence(x, batch_first=True)

        h0 = torch.zeros(d_layers, len(lens_unpacked), self.hidden_dim).requires_grad_()
        # Initial cell state (zeroes)
        c0 = torch.zeros(d_layers, len(lens_unpacked), self.hidden_dim).requires_grad_()

        # get RNN outputs
        out_packed, (_, __) = self.lstm(x, (h0.detach(), c0.detach()))
        # Unpacking out
        out, l_unpacked = pad_packed_sequence(out_packed, batch_first=True)

        out = out[:, -1, :]
        out = self.fc(out)

        return out


def train_model_lstm(model, train_loader, lengths_train, optimizer, criterion, n_epochs=100, batch_size=32,
                     n_features=1, val=None, lengths_val=None, patience=None):
    ''' Train the model
    '''

    # Check if early stop is enabled:
    if patience is not None:
        # Initialize EarlyStopping
        early_stopping = EarlyStopping(patience=patience)

    n_total_steps = len(train_loader)
    loss_train = []  # store the mean loss for each epoch of training
    loss_val = []  # store the mean loss for each epoch of validation
    for epoch in range(n_epochs):
        loss_epoch = []
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.view(batch_size, -1, n_features)
            # Packing using pack padded sequence
            x_batch = pack_padded_sequence(x_batch, lengths_train[i], batch_first=True)
            # Clear gradients
            optimizer.zero_grad()

            prediction = model(x_batch)

            # Forward pass
            loss = criterion(prediction, y_batch)

            # Backward and optimize
            loss.backward()
            # Update parameters
            optimizer.step()

            loss_epoch.append(loss.item())
        loss_train.append(mean(loss_epoch))

        if epoch % 1 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {mean(loss_epoch):.4f}')

        if val is not None:
            with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
                loss_val_epoch = []
                # turn off the regularisation during evaluation
                model.eval()
                for j, (X_val, y_val) in enumerate(val):

                    X_val = X_val.view(batch_size, -1, n_features)
                    # Packing using pack padded sequence
                    X_val = pack_padded_sequence(X_val, lengths_val[j], batch_first=True)


                    preds = model(X_val)  # get net's predictions
                    loss = criterion(preds, y_val)

                    loss_val_epoch.append(loss.item())
                loss_val.append(mean(loss_val_epoch))
            if epoch % 1 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Loss (on validation): {mean(loss_val_epoch):.4f}')

            if patience is not None:
                # Check if our patience is over (validation loss increased for given steps)
                early_stopping(np.array(mean(loss_val_epoch)), model)

                if early_stopping.early_stop:
                    print('Out of Patience. Early stopping... ')
                    break

        # checks if we will go back to the checkpoint
    if patience != -1 and early_stopping.early_stop == True:
        print('Loading model from checkpoint...')
        model.load_state_dict(torch.load('checkpoint.pt'))
        print('Checkpoint loaded.')

    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(loss_train) + 1), loss_train, label='Training Loss')
    plt.plot(range(1, len(loss_val) + 1), loss_val, label='Validation Loss')

    # find position of lowest validation loss
    minposs = loss_val.index(min(loss_val)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, max(loss_val + loss_train))  # consistent scale
    plt.xlim(0, len(loss_train) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model


def predict_model_lstm(model, test_loader, lengths_test, batch_size, n_features, criteriion):
    # Make predictions using model
    preds = []
    true_values = []
    loss = 0
    model.eval()  # prep model for evaluation

    for i, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.view([batch_size, -1, n_features])
        # Packing using pack padded sequence
        x_batch = pack_padded_sequence(x_batch, lengths_test[i], batch_first=True)
        # Make predictions
        pred = model(x_batch)
        preds.append(np.argmax(pred.detach().numpy(), axis=1)[0])
        true_values.append(y_batch.detach().numpy()[0])
        loss += criteriion(pred, y_batch)

    #Calculate Accuracy
    accuracy = sum(np.array(preds) == np.array(true_values))/len(true_values)
    return preds, true_values, accuracy


def length_paddable(length, batch_size):
    """ Makes Tensor usable for pack padded sequence
    :param length: Tensor
    :param batch_size: int
    :return: paddable List of sequence lengths of each batch element (must be on the CPU if provided as a tensor).
    """
    paddable = []
    for i in range(len(length)):
        _ = []
        for j in range(batch_size):
            _.append(length[i])
        paddable.append(_)

    return paddable


