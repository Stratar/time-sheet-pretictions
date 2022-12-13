import numpy as np
from file_reader import read_file
from stattests import stat_mode_initialiser
from rnn import store_results, plot_history, run_and_plot_predictions, store_individual_losses
from preprocessing import *
from neuralnet import GRUNeuralNetwork, ConvolutionalNeuralNetwork, AdvGRUNeuralNetwork, AdvLSTMNeuralNetwork
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
multi-dimensional array for the scaler storage. 
Check individual trainings to see how scaling back works

* Convert Pandas data handling to postgres, minimise pandas operations
EMAIL FOR HELP BUILDING DATABASE IN POSTGRES

* Finish up transfer learning parallel instead of sequential models (Check how to give the inputs)
Difference between keras Input and keras InputLayer? 
tf.keras.Input shows odd output shape in model summary, but it connects properly to the model (Tensor type)
tf.keras.InputLayer has okay output shape, but does not connect properly to model (Layer type)
There may be an issue with loading and saving the model. Trying saving weights instead and loading them into the model.
This presented some issues in the past, but loading the full model doesn't work either. 
This works now, but there is no GRU connection, but only regular dense layers appended in the end.
'''


def get_savefile_name(mode, model_name, features, transfer_learning=False):
    mode_name = ''
    if mode == 1: mode_name = 'multivariate_'
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
    '''
    mode = 1
    split = 1                       # Split the data according to employees and their individual companies they work for
    individual = False               # Checks whether each worker is going to be trained separately, or all together MIGHT BE REDUNDANT AND CAN USE genenral_prediction_mode instead
    connection = False               # Enable if there is a connection to the Akyla database
    general_prediction_mode = True # Controls whether the predictions will be made for each specific worker, or general
    in_win_size = 14                # Control how many days are used for forecasting the working hours
    out_win_size = 7                # Controls how many days in advance the
    start_at = 0
    if len(sys.argv) > 1:
        mode = int(sys.argv[1])
        try:
            start_at = int(sys.argv[2])
        except Exception as e:
            print(f'Did not get start_at due to error: {e}\n'
                  f'Ignore if you are training a general prediciton model... No starting position was provided.')



    '''
    The features considered change depending on the mode, as well as depending on the kinds of data that we want to 
    use for the forecasting of working hours.
    '''
    if mode == 1: FEATURES = ['is_holiday', 'dayofweek', 'weekofyear', 'active_assignments', 'timecard_totalhours', 'timecardline_amount']
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
    # The last 10 timecards are useless, so remove them
    df_list = df_list[:-13]

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
        model = AdvLSTMNeuralNetwork(input_shape, output_shape, n_layers=16, lstm_size=128)
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
    evaluation_mode = False
    transfer_learning = False
    load_model = False
    if load_model:
        full_name = get_savefile_name(mode, '', FEATURES, transfer_learning=transfer_learning) # Get the full name for the results
        path = f'saved model weights/{full_name}'
        '''
        Get the generalised model, that's been trained on previously encountered data and extend it with a GRU layer
        for better predictions for the personalised model with few inputs. 
        '''
        base_model = AdvLSTMNeuralNetwork(input_shape, output_shape, n_layers=16, lstm_size=128)
        base_model.load(path)
        if transfer_learning:
            '''
            Instead of adding the pre-made GRU network, make a new class that connects to the end of the generalised
            LSTM network.
            '''
            base_model.disable_training()
            # model_pers = AdvGRUNeuralNetwork(input_shape, output_shape, n_layers=10, gru_size=128)
            # model_pers.compile()
            # concatenated_output_layer = tf.keras.layers.concatenate([model_base.output, model_pers.get_model().output], name="concatenated_out_layer")


            # model.build(input_shape=(None, 14, 6))
            model = tf.keras.models.Sequential([
                base_model.get_model(),
            tf.keras.layers.Dense(input_shape[1], name=f'RepeatVector_mix'),
            tf.keras.layers.Dense(output_shape[1], name='output_mix')
            ])
            model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                                        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                                        tf.keras.metrics.KLDivergence()])

            model.build(input_shape=(None, 14, 6))
            print(model.summary())

    for cnt, df_np in enumerate(df_list):
        if evaluation_mode: continue
        if (not general_prediction_mode) and (cnt < start_at):continue
        print('the size is: ', df_np.size)
        if df_np.size < 30: continue # If the input if too small, there is no point training on it

        '''
        In the case that the prediction mode is set to be on the individual, there needs to be a model specialised to 
        each flexworker, without impacting bias from other cases. Individual models should be trained and
        SAVED separately.
        '''
        if (not general_prediction_mode) and not transfer_learning:
            print("Create a new GRU model")
            # Pass an argument for saving each model separately
            model = AdvGRUNeuralNetwork(input_shape, output_shape, n_layers=8, gru_size=128)
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
            print(type(x_train))
            print(x_train.shape)
            history = model.fit(x_train, y_train, x_val, y_val, 500)
            # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250)
            # cp = tf.keras.callbacks.ModelCheckpoint('model_advgru.h5', save_best_only=True)
            # history = model.fit(x_train, y_train, shuffle=False, validation_data=[x_val, y_val], epochs=800,
            #                          callbacks=[cp, early_stop])
        except Exception as e:
            print(f"Exception thrown when trying to fit: {e}")

            continue

        if individual:
            df_res, result_values = run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test, y_test, scalers[-1])
            dict_individual_losses["train loss"].append(result_values[0])
            dict_individual_losses["value loss"].append(result_values[4])
            dict_individual_losses["test loss"].append(result_values[8])
            print(f"train loss: {dict_individual_losses['train loss'][-1]}")
            print(f"value loss: {dict_individual_losses['value loss'][-1]}")
            print(f"test loss: {dict_individual_losses['test loss'][-1]}")
        if (not general_prediction_mode) and (cnt == end_at): break

    end = datetime.now()
    train_time = (end-start).total_seconds()                                    # Get the training time of the model

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
                                                                        y_test, scalers[-1])

    store_results(hp, result_values, df_res, full_name[:-3], start_at)
