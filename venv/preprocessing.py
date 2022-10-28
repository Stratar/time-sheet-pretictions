import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math


'''
The data needs to be handled before being forwarded to the selected model. This file contains most of the functions 
necessary to format the data in such a way that is acceptable by most algorithms.

The prefix multi refers to the function's ability to handle multiple variables as input 
'''


def val_scaler(df_np):
    scaler = MinMaxScaler()
    df_np = scaler.fit_transform(df_np.reshape(-1,1)).reshape(1,-1)[0]
    return df_np, scaler


def multi_scaler(df_np):
    scalers = []
    df_np = np.transpose(df_np)
    for i in range(df_np.shape[0]):
        scaler = MinMaxScaler()
        df_np[i] = scaler.fit_transform(df_np[i].reshape(-1,1)).reshape(1,-1)[0]
        scalers.append(scaler)
    df_np = np.transpose(df_np)
    scaler = scalers[-1]
    del scalers
    return df_np, scaler


def df_to_np(df):
    df_np = df.to_numpy()
    return df_np


# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, input_sequence_length time steps per sample, and f features
def partition_dataset(data, in_win_size, out_win_size):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(in_win_size, data_len - out_win_size):
        x.append([data[i - in_win_size:i]])  # contains input_sequence_length values 0-input_sequence_length * columns
        y.append([data[i:i + out_win_size]])  # contains the prediction values for validation

    # Convert the x and y to numpy arrays
    x = np.array(x).reshape(len(x),in_win_size, 1)
    y = np.array(y).reshape(len(y),out_win_size, 1)

    return x, y


def multi_partition_dataset(data, in_win_size, out_win_size):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(in_win_size, data_len - out_win_size):
        # print(f'the current row is:\n{data[i - in_win_size:i]}')
        row = [r for r in data[i - in_win_size:i]]
        x.append(row)
        print(data.shape)
        label = [r[-1] for r in data[i:i + out_win_size]]
        y.append(label)

    n_dims = data.shape[1]
    # Convert the x and y to numpy arrays
    x = np.array(x).reshape(len(x),in_win_size, n_dims)
    y = np.array(y).reshape(len(y),out_win_size, 1)

    return x, y


def data_split(df_np, in_win_size):
    train_data_length = math.ceil(df_np.shape[0] * 0.85)
    val_data_length = math.ceil(df_np.shape[0] * 0.9)
    train_data = df_np[:train_data_length]
    val_data = df_np[train_data_length - in_win_size :val_data_length]
    test_data = df_np[val_data_length - in_win_size:]

    return  train_data, val_data, test_data
