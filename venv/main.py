from file_reader import read_file
from stattests import general_statistics
import pandas as pd
from rnn import plot_predictions, store_results, plot_history
from preprocessing import *
from neuralnet import GRUNeuralNetwork, ConvolutionalNeuralNetwork, AdvGRUNeuralNetwork, AdvLSTMNeuralNetwork, ConvLSTMNeuralNetwork
import sys
from datetime import datetime


def run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test, y_test, scaler):

    res_train, mse_train, kl_train, acc_train, rmse_train, mae_train = plot_predictions(model, x_train, y_train, scaler)
    print(mse_train, kl_train, acc_train, rmse_train, mae_train)
    res_val, mse_val, kl_val, acc_val, rmse_val, mae_val = plot_predictions(model, x_val, y_val, scaler)
    print(mse_val, kl_val, acc_val, rmse_val, mae_val)
    res_test, mse_test, kl_test, acc_test, rmse_test, mae_test = plot_predictions(model, x_test, y_test, scaler)
    print(mse_test, kl_test, acc_test, rmse_test, mae_test)

    df_res = pd.concat([res_train, res_val, res_test], axis=1)
    result_values = [mse_train, kl_train, acc_train, rmse_train, mae_train, mse_val, kl_val, acc_val, rmse_val, mae_val,
                     mse_test, kl_test, acc_test, rmse_test, mae_test]
    return df_res, result_values


if __name__ == '__main__':
    mode = 0
    split = 1
    individual = True # Checks whether each worker is going to be trained separately, or all togeter
    connection = False
    if len(sys.argv) > 1:
        mode = int(sys.argv[1])
    # A way of making the program swap between univariate and multivariate approaches. 0: univariate, 1: multivariate

    in_win_size = 7
    out_win_size = 1
    # FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'assignment_flexworkerid', 'timecardline_amount']
    if mode == 1: FEATURES = ['dayofweek', 'dayofyear', 'weekofyear', 'timecardline_amount']
    if mode == 0: FEATURES = ['timecardline_amount']

    # M_FEATURES = ['dayofyear', 'dayofweek', 'year', 'staffingcustomer_companyname',
    #             'assignment_flexworkerid', 'timecardline_amount']

    '''
    The file is read from either a locally stored excel file, or through a query to the remote database. According
    to the selected query arguments, the dataframe will contain columns selected based on initial speculation.
    The return will be a dataframe if no argument is given to the function. If the argument 1 is passed, the dataset
    will be split into a number of sub dataframes contained in a list. Therefore if the argument is 1, the return will
    be a list.
    '''
    df = read_file(connection=connection, store_locally=False)
    if mode==2:
        FEATURES = ['assignment_startdate', 'assignment_enddate', 'quarter', 'weekofyear', 'assignmentcomponent_startdate',
                    'assignmentcomponent_enddate', 'assignment_flexworkerid', 'staffingcustomer_companyname',
                    'timecardline_amount']
        print("Using statistic analysis mode")
        print(f"Number of workers considered: {len(df[FEATURES[-3]].unique())}")
        print(f"Number of companies considered: {len(df[FEATURES[-2]].unique())}")
        # general_statistics(df)
        df_list = create_subsets(df, FEATURES, split=split)
        i = 27 # Change this to start the loop sooner or later
        for cnt, df in enumerate(df_list):
            if cnt < i: continue
            print(f"\n**********************************************\n"
                  f"            Input number {cnt+1}:"
                  f"\n**********************************************")
            print(f"Worker: {df[FEATURES[0]][0]}")
            print(f"Company: {df[FEATURES[1]][0]}")
            general_statistics(df)
        exit()
    df_list = create_subsets(df, FEATURES, split=split)
    # df_list = fill_gaps(df_list)
    df_list = fill_gaps(df_list, dt_inputs=True)
    df_list = uniform_data_types(df_list)
    df_list, scaler = convert_and_scale(df_list)
    df_list, test_data = get_test_set_np(df_list, in_win_size=in_win_size)
    df_list, val_data = get_test_set_np(df_list, in_win_size=in_win_size)

    if mode == 0:
        x_test, y_test = partition_dataset(test_data, in_win_size, out_win_size)
        x_val, y_val = partition_dataset(val_data, in_win_size, out_win_size)
    elif mode == 1:
        x_test, y_test = multi_partition_dataset(test_data, in_win_size, out_win_size)
        x_val, y_val = multi_partition_dataset(val_data, in_win_size, out_win_size)

    '''
    After the first round of assembling the necessary data for the model to work, they need to be further refined by
    performing a series of statistical tests that will narrow the list down even further.
    df = general_statistics(df)

    After the statistics have been collected in the traditional way, on the original data format (not the raw one from
    importing from the database), it can then be embedded into a different format in order to make it more accessible
    for a neural network to work with. This process requires the embeddings to be generated through an entirely
    separate embedding training network.

    Use lists of the features, and target variables from the data collection.
    '''
    '''
    The data should be formatted as a list of dataframes that will be preprocessed one-by-one 
    '''
    x = lambda n: n.shape[1:] if (len(n.shape) > 1) else [1]
    input_shape = x_test.shape[1:]
    output_shape = x(y_test)

    model = AdvGRUNeuralNetwork(input_shape, output_shape)
    model.compile()
    model.check()

    start = datetime.now()
    i = 26
    dict_individual_losses = {"train loss":[], "value loss":[], "test loss": []}
    for cnt, df_np in enumerate(df_list):
        if cnt < i:continue

        print(f"\n********************************************************************************************\n"
            f"                                        Iteration {i}:"
            f"\n********************************************************************************************")
        i += 1

        if mode == 0:
            x_train, y_train = partition_dataset(df_np, in_win_size, out_win_size)

        elif mode == 1:
            x_train, y_train = multi_partition_dataset(df_np, in_win_size, out_win_size)

        # print(y_train.shape)

        try:
            history = model.fit(x_train, y_train, x_val, y_val, 80)
        except Exception as e:
            print(f"Exception thrown: {e}")
            continue

        if individual:
            df_res, result_values = run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test, y_test, scaler)
            dict_individual_losses["train loss"].append(result_values[0])
            dict_individual_losses["value loss"].append(result_values[5])
            dict_individual_losses["test loss"].append(result_values[10])
            print(f"train loss: {dict_individual_losses['train loss'][-1]}")
            print(f"value loss: {dict_individual_losses['value loss'][-1]}")
            print(f"test loss: {dict_individual_losses['test loss'][-1]}")

    if individual:

        df_iterative = pd.DataFrame.from_dict(dict_individual_losses)
        df_iterative["train mean"] = df_iterative["train loss"].mean()
        df_iterative["value mean"] = df_iterative["value loss"].mean()
        df_iterative["test mean"] = df_iterative["test loss"].mean()
        df_iterative.to_excel(f"../../data/results/RNN/losses.xlsx")


    end = datetime.now()
    train_time = (end-start).total_seconds()
    # plot_history(history)

    hp = [train_time, model.n_layers, input_shape, output_shape, model.layer_size, model.hid_size, model.lr, model.epochs]
    '''
    Uncomment the first four lines in order to run predictions on the test and validation sets as well.
    '''
    if not individual:
        df_res, result_values = run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test, y_test, scaler)

    mode_name = ''
    if mode == 1: mode_name = 'multivariate_'
    else: mode_name = 'univariate_'
    model_name = model.name + "_"
    target_name = FEATURES[-1]

    store_results(hp, result_values, df_res, mode_name+model_name+target_name)
