from file_reader import read_file
from stattests import general_statistics
import pandas as pd
from encoder import encode_data_keras2
from neuralnet import NeuralNet
from rnn import run_rnn, plot_predictions, store_results
from preprocessing import *
from boosted_tree import start_boosted_tree
import sys


def predict(df):
    # Get the independent variables ready
    #X = pd.get_dummies(data=df.loc[:,df.columns !='totalhours'])
    #X = pd.get_dummies(data=df.drop(['amount', 'totalhours', 'totalexpense'], axis=1))
    X = df[['period', 'companyname']]
    X = X.astype(str)
    #X = tf.convert_to_tensor(X)
    Y  = df[['totalhours']]
    #Y = tf.convert_to_tensor(df['amount'])

    encode_data_keras2(X, Y)
    #use_nn(X,Y)


def use_nn(x, y):

    input_shape = x[1,:].shape
    output_size = 1
    nn = NeuralNet(input_shape, output_size, 10, 350, 0.05)
    nn.compile()
    nn.fit(x, y)
    result = nn.evaluate(x, y)
    print(result)


def get_X_y(df, in_win_size=5, out_win_size=6):
    '''Create x and y variables based on the time-series data.'''
    df_np = df.to_numpy()
    x = []
    y = []
    scaler = MinMaxScaler()
    df_np = scaler.fit_transform(df_np.reshape(-1,1)).reshape(1,-1)[0]
    for i in range(len(df_np)-(in_win_size + out_win_size - 1)):
        row = [[a] for a in df_np[i:i+in_win_size]]
        x.append(row)
        if out_win_size == 1:
            label =df_np[i+in_win_size]
        else:
            label = [[b] for b in df_np[i + in_win_size:i + in_win_size + out_win_size]]
        y.append(label)
    return np.array(x), np.array(y)


if __name__ == '__main__':

    # A way of making the program swap between univariate and multivariate approaches. 0: univariate, 1: multivariate
    mode = 1
    split = 0
    n_var_name = ""
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    '''
    The file is read from either a locally stored excel file, or through a query to the remote database. According
    to the selected query arguments, the dataframe will contain columns selected based on initial speculation.
    The return will be a dataframe if no argument is given to the function. If the argument 1 is passed, the dataset
    will be split into a number of sub dataframes contained in a list. Therefore if the argument is 1, the return will
    be a list.
    '''
    df = read_file(split=split)

    # After the first round of assembling the necessary data for the model to work, they need to be further refined by
    # performing a series of statistical tests that will narrow the list down even further.
    # df = general_statistics(df)

    # After the statistics have been collected in the traditional way, on the original data format (not the raw one from
    # importing from the database), it can then be embedded into a different format in order to make it more accessible
    # for a neural network to work with. This process requires the embeddings to be generated through an entirely
    # separate embedding training network.

    # Use lists of the features, and target variables from the data collection.

    in_win_size = 7
    out_win_size = 1

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'amount']
    TARGET = FEATURES[-1]

    '''
    Boosted tree method requires slightly different approach, so it is separated from the rest.
    IT DOES NOT WORK YET!!
    '''
    # start_boosted_tree(df[FEATURES], in_win_size, TARGET)
    '''End of boosted tree section'''

    '''Start new data preprocessing'''
    if mode == 0:
        # For univariate predictions
        if split == 1:
            df_np = []
            for worker_data in df:
                d_np = df_to_np(worker_data[TARGET])  # [:7939]
                d_np, scaler = val_scaler(d_np)
                df_np.append(d_np)
        else:
            df_np = df_to_np(df[TARGET]) #[:7939]
            df_np, scaler = val_scaler(df_np)
    elif mode == 1:
        # For multivariate predictions
        if split == 1:
            df_np = []
            for worker_data in df:
                d_np = df_to_np(worker_data[TARGET])  # [:7939]
                d_np, scaler = multi_scaler(d_np)
                df_np.append(d_np)
        else:
            df_np = df_to_np(df[FEATURES]) #[:7939]
            df_np, scaler = multi_scaler(df_np)
        n_var_name = "_multi"

    #df_np, scaler = scaler(df_np)

    # Split the training data into train and train data sets
    # As a first step, we get the number of rows to train the model on 80% of the data
    # for worker_data in df_list:
    if split == 1:
        train_data, val_data, test_data = [], [], []
        for worker_data in df_np:
            tr_data, v_data, te_data = data_split(worker_data, in_win_size)
            train_data.append(tr_data)
            val_data.append(v_data)
            test_data.append(te_data)
    else:
        train_data, val_data, test_data = data_split(df_np, in_win_size)

    # Split the train, val and test data to x and y sets
    if mode == 0:
        if split == 1:
            x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
            for w_train_data, w_val_data, w_test_data in zip(train_data, val_data, test_data):
                w_x_train, w_y_train = partition_dataset(w_train_data, in_win_size, out_win_size)
                w_x_val, w_y_val = partition_dataset(w_val_data, in_win_size, out_win_size)
                w_x_test, w_y_test = partition_dataset(w_test_data, in_win_size, out_win_size)
                x_train.append(w_x_train)
                y_train.append(w_y_train)
                x_val.append(w_x_val)
                y_val.append(w_y_val)
                x_test.append(w_x_test)
                y_test.append(w_y_test)
        else:
            x_train, y_train = partition_dataset(train_data, in_win_size, out_win_size)
            x_val, y_val = partition_dataset(val_data, in_win_size, out_win_size)
            x_test, y_test = partition_dataset(test_data, in_win_size, out_win_size)
    elif mode == 1:
        if split == 1:
            x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
            for w_train_data, w_val_data, w_test_data in zip(train_data, val_data, test_data):
                w_x_train, w_y_train = multi_partition_dataset(w_train_data, in_win_size, out_win_size)
                w_x_val, w_y_val = multi_partition_dataset(w_val_data, in_win_size, out_win_size)
                w_x_test, w_y_test = multi_partition_dataset(w_test_data, in_win_size, out_win_size)
                x_train.append(w_x_train)
                y_train.append(w_y_train)
                x_val.append(w_x_val)
                y_val.append(w_y_val)
                x_test.append(w_x_test)
                y_test.append(w_y_test)
        # For the multivariate input predictions
        else:
            x_train, y_train = multi_partition_dataset(train_data, in_win_size, out_win_size)
            x_val, y_val = multi_partition_dataset(val_data, in_win_size, out_win_size)
            x_test, y_test = multi_partition_dataset(test_data, in_win_size, out_win_size)

    # Train the model on the train data and evaluate it with the val data
    model, hp = run_rnn(x_train, y_train, x_val, y_val, split=split)

    res_train, mse_train, kl_train, acc_train, rmse_train, mae_train = plot_predictions(model, x_train, y_train, scaler)
    print(mse_train, kl_train, acc_train, rmse_train, mae_train)
    res_val, mse_val, kl_val, acc_val, rmse_val, mae_val = plot_predictions(model, x_val, y_val, scaler)
    print(mse_val, kl_val, acc_val, rmse_val, mae_val)
    res_test, mse_test, kl_test, acc_test, rmse_test, mae_test = plot_predictions(model, x_test, y_test, scaler)
    print(mse_test, kl_test, acc_test, rmse_test, mae_test)
    df_res = pd.concat([res_train, res_val, res_test], axis=1)
    result_values = [mse_train, kl_train, acc_train, rmse_train, mae_train, mse_val, kl_val, acc_val, rmse_val, mae_val,
                     mse_test, kl_test, acc_test, rmse_test, mae_test]

    store_results(hp, result_values, df_res, [TARGET, str(model._name), n_var_name])
    '''End new data preprocessing'''
