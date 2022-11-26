import random
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import lib
import numpy as np

'''Create 10-point sequences of a sine and a cosine with frequency f = 40 Hz. The purpose is to predict the
cosine given the sine sequence. Choose a constant and small distance between successive points.
Train a Recurrent Neural Network (RNN), which will accept as input the sine sequences and should
predict the corresponding cosine sequences. Instead of using the vanilla RNN, LSTMs and GRUs can also
be used.
Comment on the differences between LSTMs/GRUs and vanilla RNNs, and comment on why these
variants are so popular'''


def point_sequence(n_points, f):
    # random sequence of 10 points with the same distance between points
    points = np.zeros(n_points)
    points[0] = random.uniform(0, 1/f)

    step = 1/(f*n_points)
    for i in range(1, n_points):
        points[i] = points[i - 1] + step
    return points


def make_data(f, n_points, n_samples):
    # Makes sequences of n_points using sin and cosin functions
    times = point_sequence(n_points, f)
    X = np.sin(2 * np.pi * f * times)
    y = np.cos(2 * np.pi * f * times)
    for i in range(n_samples-1):
        times = point_sequence(n_points, f)
        X = np.vstack((X, np.sin(2 * np.pi * f * times)))
        y = np.vstack((y, np.cos(2 * np.pi * f * times)))
    return X, y


X, y = make_data(40, 10, 1000)  # Make the data using 10 points and 1000 samples

# splitting for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=False)

# Vanilla Rnn model
rnn = lib.RNN(input_size=X_train.shape[1], output_size=y_train.shape[1], hidden_dim=20, n_layers=1)
# LSTM Rnn model
lstm = lib.RNN_LSTM(input_size=X_train.shape[1], output_size=y_train.shape[1], hidden_dim=20, n_layers=1)
# GRU Rnn model
gru = lib.RNN_GRU(input_size=X_train.shape[1], output_size=y_train.shape[1], hidden_dim=20, n_layers=1)

criterion = nn.MSELoss()  # Criterion to be used by all those 3 models.

batch_size = 64  # Seting the batch size

X_train_t, y_train_t = torch.Tensor(X_train), torch.Tensor(y_train)  # Convert to Tensor
X_test_t, y_test_t = torch.Tensor(X_test), torch.Tensor(y_test)  # Convert to Tensor

train = TensorDataset(X_train_t, y_train_t)  # Combine them
test = TensorDataset(X_test_t, y_test_t)  # Combine them

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)  # Dataloader train data
test_loader = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)  # Dataloader test data

print('--------------------Training rnn----------------------')
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
rnn = lib.train_model(rnn, train_loader, optimizer, criterion, n_epochs=20, batch_size=batch_size, n_features=X_train.shape[1])
print('--------------------Training lstm----------------------')
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
lstm = lib.train_model(lstm, train_loader, optimizer, criterion, n_epochs=20, batch_size=batch_size, n_features=X_train.shape[1])
print('--------------------Training gru----------------------')
optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
gru = lib.train_model(gru, train_loader, optimizer, criterion, n_epochs=20, batch_size=batch_size, n_features=X_train.shape[1])

# Make predictions using the RNN model
preds_r, true_values = lib.predict_model(rnn, test_loader, batch_size=1, n_features=X_train.shape[1])
# Make predictions using the RNN LSTM model
preds_l, _ = lib.predict_model(lstm, test_loader, batch_size=1, n_features=X_train.shape[1])
# Make predictions using the RNN GRU model
preds_g, _ = lib.predict_model(gru, test_loader, batch_size=1, n_features=X_train.shape[1])

# Scatter predictions and true values for each model
fig, ax = plt.subplots(1, 3, figsize=(16, 8))
ax[0].scatter(np.linspace(0, 1, 10), preds_r[1])
ax[0].scatter(np.linspace(0, 1, 10), true_values[1])
ax[0].set_title('Vanilla RNN')

ax[1].scatter(np.linspace(0, 1, 10), preds_l[5])
ax[1].scatter(np.linspace(0, 1, 10), true_values[5])
ax[1].set_title('LSTM RNN')

ax[2].scatter(np.linspace(0, 1, 10), preds_g[69])
ax[2].scatter(np.linspace(0, 1, 10), true_values[69])
ax[2].set_title('GRU RNN')
fig.suptitle('Predictions and True values (Normalized)')
plt.show()
