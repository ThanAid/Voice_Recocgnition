import os
from glob import glob
import librosa
import re
import matplotlib.colors as mcolors
import numpy as np

from matplotlib import pyplot as plt


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


def extract_features(y, window=25, step=10, n_mfcc=13, Fs=16000):
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

    for i in range(len(y)):
        mfccs.append(librosa.feature.mfcc(y=y[i], sr=16000, n_mfcc=n_mfcc, n_fft=window, hop_length=window - step).T)
        delta1.append(librosa.feature.delta(data=mfccs[i], order=1))
        delta2.append(librosa.feature.delta(data=mfccs[i], order=2))

    print('\nFeature extraction completed for all the data.')

    return mfccs, delta1, delta2


def plot_hist(mfcc, digits, name_list):
    """Plots histogram of given digits
       Args:
           mfcc (list)
           digits (list): list containing digits (str) you want to plot (if you want more than 2 digits it needs changes)
           name_list (list): List containing the names of the files.
       """

    def find_indices(list_to_check, item_to_find):
        # Find the indices of a specific value on a list
        indices = []
        for idx, value in enumerate(list_to_check):
            if value == item_to_find:
                indices.append(idx)
        return indices

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
