from keras.metrics import RootMeanSquaredError, KLDivergence, Accuracy, MeanAbsoluteError
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from result_plotter import table_display

dir_path = os.path.dirname(os.path.realpath(__file__))


'''Some things to note about the GRU networks: 
They can utilise the node dropout, allowing them to avoid overfitting from nodes in the input or recurrent layers.
The dropout parameter controls the fraction of units dropped for the linear transformation of the inputs.
The recurrent dropout is also a fraction of the units dropped for the linear transformation of the recurrent state.
The default activation of the GRU layer is tanh, which produces values in the range of -1 to 1.
The default recurrent activation is a sigmoid function, producing values between 0 and 1. 

'''


def scale_data(d):
    scaler = MinMaxScaler()
    scaler.fit(d)
    d = scaler.transform(d)
    return d


from sklearn.metrics import mean_squared_error as mse


def res_dataframe(res, y):
    df = pd.DataFrame(columns=['Predictions', 'Actuals'])
    df['Predictions'] = res
    y = y.T
    df['day'] = y[0].flatten().astype(int)
    df['week'] = y[1].flatten().astype(int)
    df['Actuals'] = y[2].flatten()
    return df


def plot_predictions(model, X, y, scalers, start=0, end=150, show_plots=True):
    time_shape = y.shape[1]
    df = res_dataframe(model.predict(X).flatten(), y) # Could remove the flatten
    # df = res_dataframe(model.predict(X).flatten(), y.flatten()) # Could remove the flatten
    y.T[-1] = scalers[-1][-1].inverse_transform(np.array(y.T[-1].flatten()).reshape(-1,1)).reshape(time_shape,-1)
    y.T[0] = scalers[-1][0].inverse_transform(np.array(y.T[0].flatten()).reshape(-1,1)).reshape(time_shape,-1)
    y.T[1] = scalers[-1][1].inverse_transform(np.array(y.T[1].flatten()).reshape(-1,1)).reshape(time_shape,-1)

    # Scaling the predictions seems problematic
    predictions = scalers[-1][-1].inverse_transform(model.predict(X).reshape(-1,1)).flatten() # Try to scale with x scaler
    df = res_dataframe(predictions, y)
    if show_plots:
        plt.plot(df['Predictions'][start:end])
        plt.plot(df['Actuals'][start:end])
        plt.title('Predictions vs Actuals')
        plt.ylabel('value')
        plt.xlabel('epoch')
        plt.legend(['Predictions', 'Actuals'], loc='upper left')
        plt.grid()
        plt.show()
    kl = KLDivergence()
    kl.update_state(scale_data(y.T[-1].reshape(-1,1)), scale_data(predictions.reshape(-1,1)))
    rmse = RootMeanSquaredError()
    rmse.update_state(scale_data(y.T[-1].reshape(-1,1)), scale_data(predictions.reshape(-1,1)))
    mae = MeanAbsoluteError()
    mae.update_state(scale_data(y.T[-1].reshape(-1,1)), scale_data(predictions.reshape(-1,1)))
    return df, mse(y.T[-1].flatten(), predictions), kl.result().numpy(), rmse.result().numpy(), mae.result().numpy()


def plot_history(history):
    # summarize history for root mean squared error
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('model rmse')
    plt.ylabel('root mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()


def table_export(df, full_name):
    # df = df.reset_index(drop=True)
    # df.loc[:,["Predictions_3"]].dropna(inplace=True)
    df.loc[:, 'flexworkerid'] = df.loc[0, 'flexworkerid']
    df.loc[:, 'staffingcustomerid'] = df.loc[0, 'staffingcustomerid']
    df.loc[:, 'Predictions_3'] = df.loc[:, 'Predictions_3'].apply(lambda x:x*2).round().apply(lambda x:x/2)
    if not os.path.isdir(f"../../data/results/RNN/{full_name}"):
        os.makedirs(f"../../data/results/RNN/{full_name}")
    df.to_csv(f"../../data/results/RNN/{full_name}/predictions_{df.loc[0, 'flexworkerid']}_{df.loc[0, 'staffingcustomerid']}.csv")


def store_results(hyperparams, history, results, full_name, mode, iteration_number=0): #Remove the iteration number
    df = pd.DataFrame(columns=['train time', 'layer number', 'input shape', 'output shape', 'layer size',
                               'hidden size', 'learning rate', 'epochs', 'flexworkerid', 'staffingcustomerid',
                               'mse train', 'kl train', 'rmse train', 'mae train', 'mse val', 'kl val', 'rmse val',
                               'mae val', 'mse test', 'kl test', 'rmse test', 'mae test'])
    list = hyperparams + history

    for col, i in zip(df, list):
        df[col] = [i]

    table_export(pd.concat([df['flexworkerid'], df['staffingcustomerid'], results[["Predictions_3", "day_3", "week_3"]]], axis=1), full_name)

    df = pd.concat([df, results], axis=1)
    if mode == 1:
        table_display(df)
        exit()
    if not os.path.isdir(f"../../data/results/RNN/{full_name}"):
        os.makedirs(f"../../data/results/RNN/{full_name}")
    df.to_excel(f"../../data/results/RNN/{full_name}/{iteration_number}v.xlsx")
    print(f"Stored as: ../../data/results/RNN/{full_name}/{iteration_number}v.xlsx")


def run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test, y_test, scalers):

    res_train, mse_train, kl_train, rmse_train, mae_train = plot_predictions(model, x_train, y_train, scalers[0:2])
    print(mse_train, kl_train, rmse_train, mae_train)
    res_val, mse_val, kl_val, rmse_val, mae_val = plot_predictions(model, x_val, y_val, scalers[2:4])
    print(mse_val, kl_val, rmse_val, mae_val)
    res_test, mse_test, kl_test, rmse_test, mae_test = plot_predictions(model, x_test, y_test, scalers[4:])
    print(mse_test, kl_test, rmse_test, mae_test)

    df_res = pd.concat([res_train, res_val, res_test], axis=1)
    cols = []
    col_dict = {"Predictions": 1, "Actuals": 1, "day": 1, "week": 1}
    for column in df_res.columns:
        if column in col_dict.keys():
            cols.append(f'{column}_{col_dict[column]}')
            col_dict[column] += 1
            continue
        cols.append(column)
    df_res.columns = cols
    result_values = [mse_train, kl_train, rmse_train, mae_train, mse_val, kl_val, rmse_val, mae_val,
                     mse_test, kl_test, rmse_test, mae_test]
    return df_res, result_values