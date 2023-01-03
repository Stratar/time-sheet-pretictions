import numpy as np
from file_reader import read_file, store_flexworkers, store_staffingcustomers,store_flex_staff_table, flex_staff_pairs_from_csv
from stattests import stat_mode_initialiser
from rnn import store_results, plot_history, run_and_plot_predictions, store_individual_losses
from preprocessing import *
from neuralnet import GRUNeuralNetwork, ConvolutionalNeuralNetwork, AdvGRUNeuralNetwork, AdvLSTMNeuralNetwork, TransferNeuralNetwork
import sys
from datetime import datetime
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
'''
TODO:
* Visualise the results better, so that you can tell what the previous weeks vs the predicted week are.
store the training values in a dictionary of dates
Fix the scaler for each training, make sure the right scaler is used for the right scenario. Could make a
multi-dimensional array for the scaler storage. Scaled values need to be converted back to original shape after
conversion. Scaled prediction values are not wrong, the model simply messes up in its predictions
Check individual trainings to see how scaling back works
Add the dates to the final export 
DO IT IN THE FORMAT OF E-UUR: Consider a table where the rows are the days of the week that's being predicted, and the 
column next to that shows the actual hours worked that day, with the predicted next to it. 
Table is okay for a start

* The input window size can be reduced from 2 weeks to 1 week, or 10 days

* Check the way that the results are stored in order to add the dayofweek and weekofyear as part of the stored output

* Experiment for better predictions:
    - Batch sizes
    - Layers (Too many layers may be counter-productive): 3 - are okay More usually lead to zero results
    - Nodes: 1000 performs more or less the same as 800 
    - Learning Rates
'''


def get_savefile_name(mode, model_name, features, transfer_learning=False):
    mode_name = ''
    if mode == 1: mode_name = 'multivariate_'
    elif mode == 3: mode_name = 'multivariate_general_'
    else: mode_name = 'univariate_'
    if transfer_learning: model_name = "AdvLSTM_"
    else: model_name = model_name + "_"
    target_name = "timecardline_amount"
    full_name = mode_name+model_name+target_name + '.h5'
    return full_name


if __name__ == '__main__':
    '''
    A way of making the program swap between univariate and multivariate approaches. 0: univariate, 1: multivariate
    and 2: statistic test mode.
    The mode is irrelevant now, what needs to be done is input the flexworkerid and staffingcustomerid that the prediction should be made for
    '''
    mode = 1
    split = 1                       # Split the data according to employees and their individual companies they work for
    individual = False               # Checks whether each worker is going to be trained separately, or all together MIGHT BE REDUNDANT AND CAN USE genenral_prediction_mode instead
    connection = True               # Enable if there is a connection to the Akyla database
    general_prediction_mode = False # Controls whether the predictions will be made for each specific worker, or general
    in_win_size = 14                # Control how many days are used for forecasting the working hours
    out_win_size = 7                # Controls how many days in advance the
    start_at = 0

    # store_flex_staff_table()

    '''
    This may need to be more compact!
    '''
    index = -1
    if len(sys.argv) > 1:
        if len(sys.argv) == 3:
            index = int(sys.argv[2])
            mode = int(sys.argv[1])
            df_fs = flex_staff_pairs_from_csv()
            flexworkerid = int(df_fs.iloc[index, 0])
            staffingcustomerid = int(df_fs.iloc[index, 1])
        elif len(sys.argv) == 2:
            # Thhis looks innefficient
            general_prediction_mode = True
            df_fs = flex_staff_pairs_from_csv()
            flexworkerid, staffingcustomerid = [], []
            for _, row in df_fs.iterrows():
                flexworkerid.append(row[0])
                staffingcustomerid.append(row[1])
        elif mode == 2 or mode == 3:
            flexworkerid = int(sys.argv[2])
            staffingcustomerid = int(sys.argv[3])
        else:
            try:
                start_at = int(sys.argv[2])
            except Exception as e:
                print(f'Did not get start_at due to error: {e}\n'
                      f'Ignore if you are training a general prediciton model... No starting position was provided.')


    if mode == 1 or mode == 3: FEATURES = ['is_holiday', 'dayofweek', 'weekofyear', 'active_assignments', 'timecard_totalhours', 'timecardline_amount']
    if mode == 0: FEATURES = ['timecardline_amount']


    df = read_file(mode, [flexworkerid, staffingcustomerid], general_prediction_mode=general_prediction_mode,
                   connection=connection, store_locally=False)
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
    # The last 10 timecards are useless, so remove them
    # df_list = df_list[:-13]
    print(f"The current list is:\n{df_list}")

    '''
    This method is currently trying to create a generalised model that trains and evaluates the model after each
    individual employee input, based on some completely new employee, in this case the last and second-last inputs. 
    
    In order to create employee specific models, the algorithm can instead train and evaluate after each epoch, but
    test and evaluate based on the future predictions of the employee that the model has been trained on.
    This creates several models trained per employee, where personalised predictions can be made. 
    '''
    input_shape = (in_win_size, len(FEATURES))
    output_shape = (out_win_size, 1) # The second value reflects how many different variables will be predicted]
    if general_prediction_mode:
        '''
        Could add scaling here instead for the whole dataset, as that produces off scalings
        '''
        df_list, x_test, y_test, x_val, y_val = data_split(df_list, in_win_size, out_win_size, mode)
        model = AdvLSTMNeuralNetwork(input_shape, output_shape, n_layers=8, lstm_size=150) # 16 128
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
    dict_individual_losses = {"train loss": [], "value loss": [], "test loss": []}
    evaluation_mode = False
    load_model = False
    transfer_learning = False
    if load_model and transfer_learning:
        full_name = get_savefile_name(mode, '', FEATURES, transfer_learning=transfer_learning) # Get the full name for the results
        path = f'saved model weights/{full_name}'
        '''
        Get the generalised model, that's been trained on previously encountered data and extend it with a GRU layer
        for better predictions for the personalised model with few inputs. 
        '''
        model = TransferNeuralNetwork(input_shape, output_shape, path, n_layers=16, lstm_size=128)
        model.compile()
        model.build()
        model.check()

    for cnt, df_np in enumerate(df_list):
        if evaluation_mode: continue # Might not be necessary
        if df_np.shape[0] < 63: continue # If the input if too small, it is not possible to train on it. 63 with 14-7 / 45 with 7-7 / 84 with 21-7
        '''
        In the case that the prediction mode is set to be on the individual, there needs to be a model specialised to 
        each flexworker, without impacting bias from other cases. Individual models should be trained and
        SAVED separately.
        '''
        if (not general_prediction_mode) and (not transfer_learning):
            print("Create a new GRU model")
            # Pass an argument for saving each model separately
            model = AdvGRUNeuralNetwork(input_shape, output_shape, n_layers=3, gru_size=450) #16-128 okay
            model.compile()
            model.check()
        print(f"\n********************************************************************************************\n"
              f"                                        Iteration {cnt}:"
              f"\n********************************************************************************************")

        if not general_prediction_mode:
            # Scale each of the data here and add them to a list of scalers to de-scale when needed
            df_np, x_test, y_test, x_val, y_val = data_split(df_np, in_win_size, out_win_size, mode)

        if mode == 0: x_train, y_train = partition_dataset(df_np, in_win_size, out_win_size)

        elif mode == 1 or mode==3: x_train, y_train = multi_partition_dataset(df_np, in_win_size, out_win_size)

        # scaler is useful to see the original data again after
        x_train, y_train, x_val, y_val, x_test, y_test, scalers = data_scaler([x_train, y_train, x_val, y_val, x_test, y_test])

        '''
        There are cases where the model fit receives empty inputs, therefore cannot properly proceed and should thus be
        skipped for now.
        In the future, this means that the current flexworker does not have enough standing data for fitting a good
        model that would be personalised to them. Some ways to counter that effect are:
            - Use the same flexworker's data from their work at different companies.
            - Apply some generalised model for forecasting.
        '''
        print(f'x train shape: {x_train.shape}')
        print(f'y train shape: {y_train.shape}')
        try:
            history = model.fit(x_train, y_train.T[-1].T, x_val, y_val.T[-1].T, 1000)
        except Exception as e:
            print(f"Exception thrown when trying to fit: {e}")
            continue

        if individual:
            df_res, result_values = run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test, y_test, scalers)
            dict_individual_losses["train loss"].append(result_values[0])
            dict_individual_losses["value loss"].append(result_values[4])
            dict_individual_losses["test loss"].append(result_values[8])
            print(f"train loss: {dict_individual_losses['train loss'][-1]}")
            print(f"value loss: {dict_individual_losses['value loss'][-1]}")
            print(f"test loss: {dict_individual_losses['test loss'][-1]}")
        if (not general_prediction_mode) and (cnt == end_at): break

    end = datetime.now()
    train_time = (end-start).total_seconds()

    if not load_model:
        full_name = get_savefile_name(mode, model.name, FEATURES)  # Get the full name for the results
        path = f'saved model weights/{full_name}'
        # from the start of the loop to finish
        model.save(path)

    if individual: store_individual_losses(dict_individual_losses, full_name, start_at)   # Store the individual loss collections

    if not evaluation_mode: plot_history(history)
    if evaluation_mode: x_train, y_train = np.zeros((7056,14,6)), np.zeros((84,14,6)) # Temporary solution
    hp = [train_time, model.n_layers, input_shape, output_shape, model.layer_size, model.hid_size, model.lr, model.epochs]

    if not individual: df_res, result_values = run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test,
                                                                        y_test, scalers)
    store_results(hp, result_values, df_res, full_name[:-3], start_at)
