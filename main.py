from matplotlib import pyplot as plt

import lib
import librosa


######################### Step 2 #####################################
wavs, y, speakers, fnames = lib.parse_free_digits('digits')
print('\nData parsing completed.')
print('---------------------------------------------------------------')

######################### Step 3 #####################################
mfccs, delta1, delta2 = lib.extract_features(wavs, window=25, step=10, n_mfcc=13, Fs=16000)
print('---------------------------------------------------------------')

######################### Step 4 #####################################
lib.plot_hist(mfccs, ['one', 'nine'], y)