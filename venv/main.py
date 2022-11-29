from file_reader import read_file
from stattests import stat_mode_initialiser
from rnn import store_results, plot_history, run_and_plot_predictions, store_individual_losses
from preprocessing import *
from neuralnet import GRUNeuralNetwork, ConvolutionalNeuralNetwork, AdvGRUNeuralNetwork, AdvLSTMNeuralNetwork
import sys
from datetime import datetime
import tensorflow as tf


def get_savefile_name(mode, model, features):
    mode_name = ''
    if mode == 1: mode_name = 'multivariate_'
    else: mode_name = 'univariate_'
    model_name = model.name + "_"
    target_name = "timecardline_amount"
    full_name = mode_name+model_name+target_name
    return full_name


if __name__ == '__main__':
    '''
    A way of making the program swap between univariate and multivariate approaches. 0: univariate, 1: multivariate
    and 2: statistic test mode.
    '''
    mode = 1
    split = 1                       # Split the data according to employees and their individual companies they work for
    individual = True               # Checks whether each worker is going to be trained separately, or all together MIGHT BE REDUNDANT AND CAN USE genenral_prediction_mode instead
    connection = False               # Enable if there is a connection to the Akyla database
    general_prediction_mode = False # Controls whether the predictions will be made for each specific worker, or general
    in_win_size = 14                 # Control how many days are used for forecasting the working hours
    out_win_size = 1                # Controls how many days in advance the

    if len(sys.argv) > 1:
        mode = int(sys.argv[1])
        try:
            start_at = int(sys.argv[2])
        except Exception as e:
            print(e)



    '''
    The features considered change depending on the mode, as well as depending on the kinds of data that we want to 
    use for the forecasting of working hours.
    '''
    if mode == 1: FEATURES = ['dayofweek', 'dayofyear', 'weekofyear', 'timecard_totalhours', 'timecardline_amount']
    if mode == 0: FEATURES = ['timecardline_amount']

    df = read_file(connection=connection, store_locally=True)

    # Takes the stat analysis path instead of trying to predict and train.
    if mode==2: stat_mode_initialiser(df, split, start_at)

    '''
    The raw data received in the df is the overall collection of the timesheet data available, with certain size
    constraints. They are groupped by flexworker and company that they work for (due to the nature of the timesheet
    representation in e-uur).
    Gaps in the timesheets, where no inputs are detected over a long time, are filled with zero (0) values, considered
    as zero hour inputs instead of blank.
    Some data types are incompatible in the way that they are stored in the database and are hence cast into similar, 
    inconsequentially different types, for the sake of compatibility.
    The data is then converted from a pandas dataframe into a numpy array and scaled between 0 and 1.     
    '''
    df_list, scalers = convert_data(df, FEATURES, split)

    '''
    This method is currently trying to create a generalised model that trains and evaluates the model after each
    individual employee input, based on some completely new employee, in this case the last and second-last inputs. 
    
    In order to create employee specific models, the algorithm can instead train and evaluate after each epoch, but
    test and evaluate based on the future predictions of the employee that the model has been trained on.
    This creates several models trained per employee, where personalised predictions can be made. 
    '''
    input_shape = (in_win_size, len(FEATURES))
    output_shape = (out_win_size, 1) # The second value reflects how many different variables will be predicted

    if general_prediction_mode:
        df_list, x_test, y_test, x_val, y_val = data_split(df_list, in_win_size, out_win_size, mode)
        model = AdvLSTMNeuralNetwork(input_shape, output_shape, lstm_size=64)
        model.compile()
        model.check()


    '''
    After the statistics have been collected in the traditional way, on the original data format (not the raw one from
    importing from the database), it can then be embedded into a different format in order to make it more accessible
    for a neural network to work with. This process requires the embeddings to be generated through an entirely
    separate embedding training network.
    '''
    start = datetime.now()
    #11, 18, 20, 21, 29, 33, 34, 36, 38, 53, 76, 79, 80, 87, 88, 93, 99, 106, 118, 119
    # Set as a means to start the training from a different index
    end_at = start_at
    dict_individual_losses = {"train loss":[], "value loss":[], "test loss": []}

    for cnt, df_np in enumerate(df_list):
        if cnt < start_at:continue

        '''
        In the case that the prediction mode is set to be on the individual, there needs to be a model specialised to 
        each flexworker, without impacting bias from other cases. Individual models should be trained and
        SAVED separately.
        '''
        if not general_prediction_mode:
            # Pass an argument for saving each model separately
            model = AdvGRUNeuralNetwork(input_shape, output_shape, n_layers=3, gru_size=128)
            model.compile()
            model.check()

        print(f"\n********************************************************************************************\n"
              f"                                        Iteration {cnt}:"
              f"\n********************************************************************************************")

        if not general_prediction_mode:
            print(f"full length: {len(df_np)}")
            df_np, x_test, y_test, x_val, y_val = data_split(df_np, in_win_size, out_win_size, mode)
            print(f"split lens:\n"
                  f"df_np: {len(df_np)}\n"
                  f"x_test: {len(x_test)}\n"
                  f"x_val: {len(x_val)}\n"
                  f"total: {len(df_np)+len(x_test)+len(x_val)}")

        if mode == 0: x_train, y_train = partition_dataset(df_np, in_win_size, out_win_size)

        elif mode == 1: x_train, y_train = multi_partition_dataset(df_np, in_win_size, out_win_size)

        '''
        There are cases where the model fit receives empty inputs, therefore cannot properly proceed and should thus be
        skipped for now.
        In the future, this means that the current flexworker does not have enough standing data for fitting a good
        model that would be personalised to them. Some ways to counter that effect are:
            - Use the same flexworker's data from their work at different companies.
            - Apply some generalised model for forecasting.
        '''
        try:
            history = model.fit(x_train, y_train, x_val, y_val, 500)
        except Exception as e:
            print(f"Exception thrown: {e}")
            continue

        if individual:
            df_res, result_values = run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test, y_test, scalers[-1])
            dict_individual_losses["train loss"].append(result_values[0])
            dict_individual_losses["value loss"].append(result_values[4])
            dict_individual_losses["test loss"].append(result_values[8])
            print(f"train loss: {dict_individual_losses['train loss'][-1]}")
            print(f"value loss: {dict_individual_losses['value loss'][-1]}")
            print(f"test loss: {dict_individual_losses['test loss'][-1]}")
        if cnt == end_at: break

    end = datetime.now()
    train_time = (end-start).total_seconds()                                    # Get the training time of the model
                                                                                # from the start of the loop to finish

    full_name = get_savefile_name(mode, model, FEATURES)                        # Get the full name for the results
                                                                                # to be saved

    if individual: store_individual_losses(dict_individual_losses, full_name, start_at)   # Store the individual loss collections

    plot_history(history)

    hp = [train_time, model.n_layers, input_shape, output_shape, model.layer_size, model.hid_size, model.lr, model.epochs]

    if not individual: df_res, result_values = run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test,
                                                                        y_test, scalers[-1])

    store_results(hp, result_values, df_res, full_name, start_at)
