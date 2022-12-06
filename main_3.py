import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import lib
import parser
import warnings

warnings.filterwarnings("ignore")  # ignore Warnings

print('--------------------------- Data parsing ---------------------------------------------')
X_train, X_test, y_train, y_test, spk_train, spk_test = parser.parser('recordings', n_mfcc=13)
print('--------------------------- Data parsing Completed------------------------------------')

######################################### Step 9 ##############################################
print('\n--------------------------- Splitting Train Data----------------------------------------')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)
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

batch_size = 1  # Setting the batch size

X_train_t = [torch.Tensor(X) for X in X_train]  # Convert to Tensor
X_val_t = [torch.Tensor(X) for X in X_val]  # Convert to Tensor
X_test_t = [torch.Tensor(X) for X in X_test]  # Convert to Tensor

# finding the length for each frame
lengths_train = []
lengths_val = []
lengths_test = []

for sample in X_train_t:
    lengths_train.append(sample.size(dim=0))
for sample in X_val_t:
    lengths_val.append(sample.size(dim=0))
for sample in X_test_t:
    lengths_test.append(sample.size(dim=0))

# sorting and getting indexes
train_ind = np.argsort((np.array(lengths_train) * -1)).tolist()  # Multiply by -1 for Decreasing order
val_ind = np.argsort(np.array(lengths_val) * -1).tolist()
test_ind = np.argsort(np.array(lengths_test) * -1).tolist()

# sorting list of tensors using the lists above
X_train_s = [X_train_t[i] for i in train_ind]
X_val_s = [X_val_t[i] for i in val_ind]
X_test_s = [X_test_t[i] for i in test_ind]

# sorting labels too
y_train_s = torch.Tensor([y_train[i] for i in train_ind]).type(torch.LongTensor)
y_val_s = torch.Tensor([y_val[i] for i in val_ind]).type(torch.LongTensor)
y_test_s = torch.Tensor([y_test[i] for i in test_ind]).type(torch.LongTensor)
#######################
X_train_p = pad_sequence(X_train_s, batch_first=True)  # Padding
X_val_p = pad_sequence(X_val_s, batch_first=True)  # Padding
X_test_p = pad_sequence(X_test_s, batch_first=True)  # Padding

train = TensorDataset(X_train_p, y_train_s)  # Combine them
val = TensorDataset(X_val_p, y_val_s)  # Combine them
test = TensorDataset(X_test_p, y_test_s)  # Combine them

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=False)  # Dataloader train data
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False)  # Dataloader val data
test_loader = DataLoader(test, batch_size=1, shuffle=False, drop_last=False)  # Dataloader test data

# Transform those lengths tensors to use via pad packed
lengths_train = torch.as_tensor(lengths_train, dtype=torch.int64)
lengths_val = torch.as_tensor(lengths_val, dtype=torch.int64)
lengths_test = torch.as_tensor(lengths_test, dtype=torch.int64)

# Using created funtion make those tensor usable by pack_padded_sequence
lengths_train = lib.length_paddable(lengths_train, batch_size)
lengths_val = lib.length_paddable(lengths_val, batch_size)
lengths_test = lib.length_paddable(lengths_test, 1)

# LSTM Rnn model
lstm = lib.LSTM(input_size=13, output_size=10, hidden_dim=60, n_layers=1, dropout=0.2, bidirectional=False)

print('--------------------Training lstm----------------------')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001, weight_decay=1e-5)  # weight_decay>0 for L2 regularization
lstm = lib.train_model_lstm(lstm, train_loader, lengths_train, optimizer, criterion, n_epochs=40, batch_size=batch_size,
                            n_features=13, val=val_loader, lengths_val=lengths_val, patience=1)
# preds, true_values, los = lib.predict_model_lstm(lstm, test_loader, 1, 13, criterion)

# Creating LSTM bidirectional
lstm_bi = lib.LSTM(input_size=13, output_size=10, hidden_dim=60, n_layers=1, dropout=0.2, bidirectional=True)

print('---------------Training lstm Bidirectional----------------')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_bi.parameters(), lr=0.001, weight_decay=1e-5)  # weight_decay>0 for L2 regularization
lstm_bi = lib.train_model_lstm(lstm_bi, train_loader, lengths_train, optimizer, criterion, n_epochs=40,
                               batch_size=batch_size,
                               n_features=13, val=val_loader, lengths_val=lengths_val, patience=4)
# preds, true_values, los = lib.predict_model_lstm(lstm, test_loader, 1, 13, criterion)
