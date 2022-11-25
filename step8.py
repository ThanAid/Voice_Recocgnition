import random
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import lib
import numpy as np


def point_sequence(n_points, f):
    # random sequence of 10 points with the same distance between points
    points = np.zeros(n_points)
    points[0] = random.random()
    step = points[0] / f
    for i in range(1, n_points):
        points[i] = points[i - 1] + step
    return points


def make_data(f, n_points, n_samples):

    X = np.sin(2 * np.pi * f * point_sequence(n_points, f))
    y = np.cos(2 * np.pi * f * point_sequence(n_points, f))
    for i in range(n_samples-1):
        X = np.vstack((X,np.sin(2 * np.pi * f * point_sequence(n_points, f))))
        y = np.vstack((y,np.cos(2 * np.pi * f * point_sequence(n_points, f))))
    return X,y


X,y=make_data(40,1,1000)
print(np.shape(X))
# splitting for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=False)
# splitting train for train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42, shuffle=False)

train_features, train_targets = torch.Tensor(X_train), torch.Tensor(y_train)
test_features, test_targets = torch.Tensor(X_test), torch.Tensor(y_test)
val_features, val_targets = torch.Tensor(X_val), torch.Tensor(y_val)

print(train_features.size(dim=1))

train = TensorDataset(train_features, train_targets)
test = TensorDataset(test_features, test_targets)
val = TensorDataset(val_features, val_targets)

batch_size=32
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)

model = lib.RNN(input_size= X_train.shape[1], hidden_size=64, num_layers=3, num_classes=y_train.shape[1])

# loss and oprimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lib.train_model(model= model, train_loader= train_loader, val_loader=val_loader, optimizer=optimizer, criterion=criterion, n_epochs=100, batch_size=32, n_features=1)
