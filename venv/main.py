import numpy as np
from file_reader import read_file, store_flexworkers, store_staffingcustomers,store_flex_staff_table, flex_staff_pairs_from_csv
from stattests import stat_mode_initialiser
from rnn import store_results, plot_history, run_and_plot_predictions
from preprocessing import *
from neuralnet import AdvGRUNeuralNetwork, AdvLSTMNeuralNetwork, TransferNeuralNetwork, SemiTransformer
import sys
from datetime import datetime
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
'''
TODO:
* Experiment for better predictions:
    - Batch sizes
    - Layers (Too many layers may be counter-productive): 3 - are okay More usually lead to zero results
    - Nodes: 1000 performs more or less the same as 800 
    - Learning Rates
    - Better pre-processing, use statistical quartile filters, mostly for median > 1, as well as outlier trimming, there
    are still cases where some days (like case 65) have more than 24 hours worked in a day.
    Consider grouping staffingcustomers together and get the number of active workers per day/week!
    
* Finish up the prediction mode 

* Clean it up!
    
    Candidates: 6(erratic), 7(variable), 8(erratic), 9(variable), 13(erratic), 14(semi-stable), 15(erratic), 17(variable)
    
        Erratic: 6, 8, 13, 15, 20(outliers), 23, 25, 36, 37, 39, 41, 61, 62, 64, 65(unrealistic), 67, 71, 72, 83,
        86, 88, 90, 93, 96, 109, 112(unrealistic), 120, 130, 138, 142, 173, 189, 204
        Variable: 7, 9, 17, 21(few), 22, 44, 50(semi), 51(semi), 52(semi), 55(semi), 59, 66, 73(unrealistic), 74,
        100, 103, 108, 113, 136(unrealistic), 137, 144, 152, 157, 165(outliers), 169(unrealistic), 184, 187, 188, 196,
        200, 202
        Semi-Stable (std > mean?): 14, 18, 27, 28(outliers), 30(outliers), 33, 46, 47, 56, 69, 101, 114, 164(outliers), 174(good), 181,
        190, 193
        Stable: 1651
        Unrealistic: 65, 73, 87, 112, 118, 125, 134, 136, 143, 151, 154, 155, 162, 165, 169, 171, 179, 181, 189, 193, 196,
        198,
        
        What do the predictions mean?
        Represent it appropriately in the presentation as table
        Measure average deviation
        
'''


def get_savefile_name(mode, model_name, fs_pair, transfer_learning=False, general_mode=True, out_win_size=1):
    mode_name = ''
    if mode == 3 or mode == 1: mode_name = 'training_'
    else: mode_name = 'other_'
    if transfer_learning: model_name = "Semi-Transformer_"
    else: model_name = model_name + "_"
    win_name = f"{out_win_size}_"
    if not general_mode:
        fw_name = f"{fs_pair[0]}_"
        sc_name = f"{fs_pair[1]}"
    else:
        fw_name = "general"
        sc_name = ""
    full_name = mode_name+model_name+win_name + fw_name + sc_name + '.h5'
    return full_name


def get_execution_mode(args):

    if len(args) > 5 or len(args) < 1:
        raise Exception("The input provided was insufficient to continue with the execution.")
        exit()

    general_training_mode = False   # Controls whether the predictions will be made for each specific worker, or general
    load_model = False              # If evaluation mode or transfer learning
    transfer_learning = False

    mode_dict = {"train": 3, "statistics": 2, "predict": 1}
    train_dict = {"general": 1, "transfer": 2}

    if args[1] in mode_dict.keys():
        mode = mode_dict[args[1]]
        if mode == 3 and args[2] in train_dict:
            if train_dict[args[2]] == 1: return mode, True, None, None, None, None
            elif train_dict[args[2]] == 2: transfer_learning, load_model = True, True
            else: raise Exception("Invalid train command given. Specify if it is \'transfer\', \'general\' or give the "
                              "corresponding flexworkerid - staffincustomerid pair.")

        if mode == 1: load_model = True

        if int(args[-1]) < 100000:
            index = int(args[-1])
            df_fs = flex_staff_pairs_from_csv()
            return mode, general_training_mode, transfer_learning, load_model, int(df_fs.iloc[index, 0]), \
                   int(df_fs.iloc[index, 1])

        fwid, scid = int(args[-2]), int(args[-1])
        return mode, general_training_mode, transfer_learning, load_model, fwid, scid

    else: raise Exception("Invalid input. Please specify the function: Train, Predict or Statistics")


if __name__ == '__main__':
    '''
    A way of making the program swap between univariate and multivariate approaches. 0: univariate, 1: multivariate
    and 2: statistic test mode.
    The mode is irrelevant now, what needs to be done is input the flexworkerid and staffingcustomerid that the prediction should be made for
    '''
    # store_flex_staff_table()
    mode, general_training_mode, transfer_learning, load_model, flexworkerid, staffingcustomerid = get_execution_mode(sys.argv)
    connection = True               # Enable if there is a connection to the Akyla database
    in_win_size = 7                # Control how many days are used for forecasting the working hours
    out_win_size = 1

    n_layers = 4 # 6 - 59.5, 2 -

    head_size = 1280 #512 - 57.87, 128 - 59.5
    num_heads = 2 #8 - 57.9, 2 - 57.87
    ff_dim = 2 #4 - 61.5, < 12, 2 - 57.9

    gru_size = 550
    lstm_size = 512


    epochs = 150 #150 - 8.5mse
    train_limit = 1200
    '''Get a list of fwids and scids when training on the whole dataset.'''
    if general_training_mode:
        df_fs = flex_staff_pairs_from_csv()
        flexworkerid, staffingcustomerid = [], []
        for _, row in df_fs.iterrows():
            flexworkerid.append(row[0])
            staffingcustomerid.append(row[1])

    start = datetime.now()
    df = read_file(mode, [flexworkerid, staffingcustomerid], general_prediction_mode=general_training_mode,
                   connection=connection, store_locally=True)
    time_postgres = datetime.now() - start
    print(f"Loaded from postgres in: {(time_postgres)/60}mins")
    if mode == 3 or mode == 1: FEATURES = ['dayofweek', 'weekofyear', 'flexworkerids',
                                           'active_assignments', 'timecardline_amount']
    # if mode == 3 or mode == 1: FEATURES = ['dayofweek', 'weekofyear', 'is_holiday', 'active_flexworkers',
    #                                        'active_assignments', 'timecard_totalhours', 'timecardline_amount']

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
    start = datetime.now()
    df_list, scalers = convert_data(df, FEATURES)

    time_preprocessing = datetime.now() - start
    print(f"Converted data in: {(time_preprocessing)/60}mins")
    '''
    This method is currently trying to create a generalised model that trains and evaluates the model after each
    individual employee input, based on some completely new employee, in this case the last and second-last inputs. 
    
    In order to create employee specific models, the algorithm can instead train and evaluate after each epoch, but
    test and evaluate based on the future predictions of the employee that the model has been trained on.
    This creates several models trained per employee, where personalised predictions can be made. 
    '''
    input_shape = (in_win_size, len(FEATURES))
    output_shape = (out_win_size, 1)

    if general_training_mode:

        df_list, x_test, y_test, x_val, y_val = data_split(df_list, in_win_size, out_win_size, mode)
        # model = AdvLSTMNeuralNetwork(input_shape, output_shape, n_layers=n_layers, lstm_size=lstm_size) # 16 128
        model = SemiTransformer(input_shape, output_shape, num_transformer_blocks=n_layers, head_size=head_size,
                                num_heads=num_heads, ff_dim=ff_dim)
        full_name = get_savefile_name(mode, model.name, [flexworkerid, staffingcustomerid], transfer_learning=transfer_learning, out_win_size=out_win_size) # Get the full name for the results
        path = f'saved model weights/{full_name}'
        model.set_path(path)
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
        full_name = get_savefile_name(mode, '', [flexworkerid, staffingcustomerid], transfer_learning=transfer_learning, out_win_size=out_win_size) # Get the full name for the results
        path = f'saved model weights/{full_name}'
        '''
        Get the generalised model, that's been trained on previously encountered data and extend it with a GRU layer
        for better predictions for the personalised model with few inputs. 
        '''
        model = TransferNeuralNetwork(input_shape, output_shape, path, n_layers=n_layers, lstm_size=lstm_size)
        model.set_path(path)
        model.compile()
        model.build()
        model.check()
    elif load_model:
        # model = AdvGRUNeuralNetwork(input_shape, output_shape, n_layers=n_layers, gru_size=gru_size)
        model = SemiTransformer(input_shape, output_shape, num_transformer_blocks=n_layers, head_size=head_size,
                                num_heads=num_heads, ff_dim=ff_dim)
        full_name = get_savefile_name(mode, model.name, [flexworkerid, staffingcustomerid], transfer_learning=transfer_learning, out_win_size=out_win_size) # Get the full name for the results
        path = f'saved model weights/{full_name}'
        model.get_model().load_weights(path)
        model.compile()
        model.check()

    # Start the timer right before the training loop.
    start = datetime.now()

    for cnt, df_np in enumerate(df_list):
        if df_np.shape[0] < 63 or np.mean(np.transpose(df_np)[-1]) < 1.75: continue # If the input if too small, it is not possible to train on it. 63 with 14-7 / 45 with 7-7 / 84 with 21-7
        if cnt > train_limit: break
        '''
        In the case that the prediction mode is set to be on the individual, there needs to be a model specialised to 
        each flexworker, without impacting bias from other cases. Individual models should be trained and
        SAVED separately.
        '''
        if (not general_training_mode) and (not transfer_learning) and not mode == 1:
            # model = AdvGRUNeuralNetwork(input_shape, output_shape, n_layers=n_layers, gru_size=gru_size) #16-128 okay, 4-450 better, 4-850

            model = SemiTransformer(input_shape, output_shape, num_transformer_blocks=n_layers, head_size=head_size,
                                    num_heads=num_heads, ff_dim=ff_dim)
            full_name = get_savefile_name(mode, model.name, [flexworkerid, staffingcustomerid], transfer_learning=transfer_learning, out_win_size=out_win_size) # Get the full name for the results
            path = f'saved model weights/{full_name}'
            model.set_path(path)
            model.compile()
            model.check()

        print(f"\n********************************************************************************************\n"
              f"                                        Iteration {cnt}:"
              f"\n********************************************************************************************")
        print(f"Time to import from postgres: {time_postgres/60} mins.\n"
              f"Time to pre-process: {time_preprocessing/60} mins")
        if not general_training_mode:
            # Scale each of the data here and add them to a list of scalers to de-scale when needed
            df_np, x_test, y_test, x_val, y_val = data_split(df_np, in_win_size, out_win_size, mode)

        if mode == 0: x_train, y_train = partition_dataset(df_np, in_win_size, out_win_size)

        elif mode == 1 or mode==3: x_train, y_train = multi_partition_dataset(df_np, in_win_size, out_win_size)

        # scaler is useful to see the original data again after
        scalers = []
        # x_train, y_train, x_val, y_val, x_test, y_test, scalers = data_scaler([x_train, y_train, x_val, y_val, x_test, y_test])

        '''
        There are cases where the model fit receives empty inputs, therefore cannot properly proceed and should thus be
        skipped for now.
        In the future, this means that the current flexworker does not have enough standing data for fitting a good
        model that would be personalised to them. Some ways to counter that effect are:
            - Use the same flexworker's data from their work at different companies.
            - Apply some generalised model for forecasting.
        '''
        if mode == 1: break
        try:
            history = model.fit(x_train, y_train.T[-1].T, x_val, y_val.T[-1].T, epochs)
            train_time = (datetime.now()-start).total_seconds()
            print(f"The train time was: {train_time/60}mins")
            if mode == 3 and not load_model: model.save(model.path)
        except Exception as e:
            print(f"Exception thrown when trying to fit: {e}")
            continue

        if len(df_list) > 1 and cnt % int(len(df_list)/2) == 0: model.set_learning_rate(model.lr/10)



    train_time = (datetime.now()-start).total_seconds()
    print(f"The train time was: {train_time/60}mins")

    if not load_model: # The weights are mostly stored when the training of the general, or personal models is complete
        full_name = get_savefile_name(mode, model.name, [flexworkerid, staffingcustomerid], out_win_size=out_win_size)  # Get the full name for the results
        path = f'saved model weights/{full_name}'
        model.save(path)

    if not (mode == 1): plot_history(history) # Not needed for the final version

    hp = [train_time, model.n_layers, input_shape, output_shape, model.layer_size, model.hid_size, model.lr, model.epochs,
          flexworkerid, staffingcustomerid]

    df_res, result_values = run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test,
                                                                        y_test, scalers)
    store_results(hp, result_values, df_res, full_name[:-3], mode)



