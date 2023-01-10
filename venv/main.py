import numpy as np
from file_reader import read_file, store_flexworkers, store_staffingcustomers,store_flex_staff_table, flex_staff_pairs_from_csv
from stattests import stat_mode_initialiser
from rnn import store_results, plot_history, run_and_plot_predictions
from preprocessing import *
from neuralnet import GRUNeuralNetwork, ConvolutionalNeuralNetwork, AdvGRUNeuralNetwork, AdvLSTMNeuralNetwork, TransferNeuralNetwork
import sys
from datetime import datetime
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
'''
TODO:
* DO IT IN THE FORMAT OF E-UUR: Consider a table where the rows are the days of the week that's being predicted, and the 
column next to that shows the actual hours worked that day, with the predicted next to it. 
Table is okay for a start
Export as a table with 5/6 columns: (scid, fwid, weekno, dayno, prediction)

* The input window size can be reduced from 2 weeks to 1 week, or 10 days

* Experiment for better predictions:
    - Batch sizes
    - Layers (Too many layers may be counter-productive): 3 - are okay More usually lead to zero results
    - Nodes: 1000 performs more or less the same as 800 
    - Learning Rates
    - Better pre-processing, use statistical quartile filters, mostly for median > 1, as well as outlier trimming, there
    are still cases where some days (like case 65) have more than 24 hours worked in a day.
    Consider grouping staffingcustomers together and get the number of active workers per day/week!
    
* Transfer learning model needs to be trained on reliable data. Train it on the new table, with as few flat values as
possible.

* Clean it up!
    
    Candidates: 6(erratic), 7(variable), 8(erratic), 9(variable), 13(erratic), 14(semi-stable), 15(erratic), 17(variable)
    
        Erratic: 6, 8, 13, 15, 20(outliers), 23, 25, 36, 37, 39, 41, 61, 62, 64, 65(unrealistic), 67, 71, 72, 83,
        86, 88, 90, 93, 96, 109, 112(unrealistic), 120, 130, 138, 142, 173, 189, 204
        Variable: 7, 9, 17, 21(few), 22, 44, 50(semi), 51(semi), 52(semi), 55(semi), 59, 66, 73(unrealistic), 74,
        100, 103, 108, 113, 136(unrealistic), 137, 144, 152, 157, 165(outliers), 169(unrealistic), 184, 187, 188, 196,
        200, 202
        Semi-Stable (std > mean?): 14, 18, 27, 28(outliers), 30(outliers), 33, 46, 47, 56, 69, 101, 114, 164(outliers), 174(good), 181,
        190, 193
        Stable:
        Unrealistic: 65, 73, 87, 112, 118, 125, 134, 136, 143, 151, 154, 155, 162, 165, 169, 171, 179, 181, 189, 193, 196,
        198,
'''


def get_savefile_name(mode, model_name, features, transfer_learning=False, out_win_size=1):
    mode_name = ''
    if mode == 1: mode_name = 'multivariate_'
    elif mode == 3: mode_name = 'multivariate_'
    else: mode_name = 'univariate_'
    if transfer_learning: model_name = "AdvLSTM_"
    else: model_name = model_name + "_"
    if out_win_size == 1: win_name = "1_"
    elif out_win_size == 7: win_name = "7_"
    target_name = "timecardline_amount"
    full_name = mode_name+model_name+win_name+target_name + '.h5'
    return full_name


def get_execution_mode(args):

    if len(args) > 5 or len(args) < 1:
        raise Exception("The input provided was insufficient to continue with the execution.")
        exit()

    general_training_mode = False # Controls whether the predictions will be made for each specific worker, or general
    load_model = False              # If evaluation mode or transfer learning
    transfer_learning = False

    mode_dict = {"train": 3, "statistics": 2, "predict": 1}
    train_dict = {"general": 1, "transfer": 2}
    if args[1] in mode_dict.keys():
        mode = mode_dict[args[i]]
        if mode == 3 and args[2] in train_dict:
            if train_dict[args[2]] == 1: return mode, True, _, _, _, _
            elif train_dict[args[2]] == 2: transfer_learning, load_model = True, True
            else: raise Exception("Invalid train command given. Specify if it is \'transfer\', \'general\' or give the "
                              "corresponding flexworkerid - staffincustomerid pair.")
        fwid = args[-2]
        scid = args[-1]
        return mode, general_training_mode, transfer_learning, load_model, fwid, scid
    else: raise Exception("Invalid input. Please specify the function: Train, Predict of Statistics")



if __name__ == '__main__':
    '''
    A way of making the program swap between univariate and multivariate approaches. 0: univariate, 1: multivariate
    and 2: statistic test mode.
    The mode is irrelevant now, what needs to be done is input the flexworkerid and staffingcustomerid that the prediction should be made for
    '''
    mode = 1
    prediction_mode = False         # Use a string input instead of this
    general_training_mode = False # Controls whether the predictions will be made for each specific worker, or general
    load_model = False              # If evaluation mode or transfer learning
    transfer_learning = False
    connection = True               # Enable if there is a connection to the Akyla database
    in_win_size = 14                # Control how many days are used for forecasting the working hours
    out_win_size = 7

    # store_flex_staff_table()
    get_execution_mode(sys.argv)

    '''
    This may need to be more compact!
    '''
    index = -1
    if len(sys.argv) > 1:
        if len(sys.argv) == 3:
            index, mode = int(sys.argv[2]), int(sys.argv[1])
            df_fs = flex_staff_pairs_from_csv()
            flexworkerid, staffingcustomerid = int(df_fs.iloc[index, 0]), int(df_fs.iloc[index, 1])
        elif len(sys.argv) == 2:
            # This looks inefficient
            general_training_mode = True
            df_fs = flex_staff_pairs_from_csv()
            flexworkerid, staffingcustomerid = [], []
            for _, row in df_fs.iterrows():
                flexworkerid.append(row[0])
                staffingcustomerid.append(row[1])
        elif mode == 2 or mode == 3:
            flexworkerid, staffingcustomerid  = int(sys.argv[2]), int(sys.argv[3])

    if mode == 3: FEATURES = ['is_holiday', 'dayofweek', 'weekofyear', 'active_assignments', 'timecard_totalhours', 'timecardline_amount']
    if mode == 0: FEATURES = ['timecardline_amount']

    df = read_file(mode, [flexworkerid, staffingcustomerid], general_prediction_mode=general_training_mode,
                   connection=connection, store_locally=False)
    # Takes the stat analysis path instead of trying to predict and train.
    if mode==2: stat_mode_initialiser(df)

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
    df_list, scalers = convert_data(df, FEATURES)

    '''
    This method is currently trying to create a generalised model that trains and evaluates the model after each
    individual employee input, based on some completely new employee, in this case the last and second-last inputs. 
    
    In order to create employee specific models, the algorithm can instead train and evaluate after each epoch, but
    test and evaluate based on the future predictions of the employee that the model has been trained on.
    This creates several models trained per employee, where personalised predictions can be made. 
    '''
    input_shape = (in_win_size, len(FEATURES))
    output_shape = (out_win_size, 1) # The second value reflects how many different variables will be predicted]
    if general_training_mode:
        '''
        Could add scaling here instead for the whole dataset, as that produces off scalings
        '''
        df_list, x_test, y_test, x_val, y_val = data_split(df_list, in_win_size, out_win_size, mode)
        model = AdvLSTMNeuralNetwork(input_shape, output_shape, n_layers=2, lstm_size=350) # 16 128
        model.compile()
        model.check()
    '''
    After the statistics have been collected in the traditional way, on the original data format (not the raw one from
    importing from the database), it can then be embedded into a different format in order to make it more accessible
    for a neural network to work with. This process requires the embeddings to be generated through an entirely
    separate embedding training network.
    '''

    dict_individual_losses = {"train loss": [], "value loss": [], "test loss": []}
    if load_model and transfer_learning:
        full_name = get_savefile_name(mode, '', FEATURES, transfer_learning=transfer_learning) # Get the full name for the results
        path = f'saved model weights/{full_name}'
        '''
        Get the generalised model, that's been trained on previously encountered data and extend it with a GRU layer
        for better predictions for the personalised model with few inputs. 
        '''
        model = TransferNeuralNetwork(input_shape, output_shape, path, n_layers=2, lstm_size=350)
        model.compile()
        model.build()
        model.check()
    # Start the timer right before the training loop.
    start = datetime.now()

    for cnt, df_np in enumerate(df_list):
        if prediction_mode or mode == 1: continue # Might not be necessary
        df_np_T = np.transpose(df_np)
        if df_np.shape[0] < 63 or np.mean(np.transpose(df_np)[-1]) < 1.5: continue # If the input if too small, it is not possible to train on it. 63 with 14-7 / 45 with 7-7 / 84 with 21-7
        '''
        In the case that the prediction mode is set to be on the individual, there needs to be a model specialised to 
        each flexworker, without impacting bias from other cases. Individual models should be trained and
        SAVED separately.
        '''
        if (not general_training_mode) and (not transfer_learning):
            model = AdvGRUNeuralNetwork(input_shape, output_shape, n_layers=6, gru_size=450) #16-128 okay
            model.compile()
            model.check()
        print(f"\n********************************************************************************************\n"
              f"                                        Iteration {cnt}:"
              f"\n********************************************************************************************")

        if not general_training_mode:
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
        try:
            history = model.fit(x_train, y_train.T[-1].T, x_val, y_val.T[-1].T, 600)
        except Exception as e:
            print(f"Exception thrown when trying to fit: {e}")
            continue

    train_time = (datetime.now()-start).total_seconds()

    # if not load_model: # The weights are mostly stored when the training of the general, or personal models is complete
    #     full_name = get_savefile_name(mode, model.name, FEATURES)  # Get the full name for the results
    #     path = f'saved model weights/{full_name}'
    #     # from the start of the loop to finish
    #     model.save(path)

    if not prediction_mode or not (mode == 1): plot_history(history) # Not needed for the final version
    if prediction_mode or  mode == 1: x_train, y_train = np.zeros((7056,14,6)), np.zeros((84,14,6)) # Temporary solution
    hp = [train_time, model.n_layers, input_shape, output_shape, model.layer_size, model.hid_size, model.lr, model.epochs]

    df_res, result_values = run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test,
                                                                        y_test, scalers)
    store_results(hp, result_values, df_res, full_name[:-3])



