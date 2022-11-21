from keras.metrics import RootMeanSquaredError, KLDivergence, Accuracy, MeanAbsoluteError
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

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


def predict(model, X):
    predictions = []
    for _ in range(10):
        if len(predictions) == 0:
            predictions = model.predict(X).flatten()
        else:
            new_prediction = model.predict(X)
            predictions = [x + y for x, y in zip(predictions, new_prediction.flatten())]
        # df = pd.DataFrame(data={'Predictions':predictions, 'Actuals':y})
    '''This line aligns the model predictions better with the actual data'''
    predictions = np.delete(np.roll(np.array(predictions) / 10, -1), -1)
    return predictions


def res_dataframe(res, y):
    df = pd.DataFrame(columns=['Predictions', 'Actuals'])
    df['Predictions'] = res
    df['Actuals'] = y
    return df


def plot_predictions(model, X, y, scaler, start=0, end=150, show_plots=False):

    predictions = scaler.inverse_transform(model.predict(X).reshape(-1,1)).reshape(1,-1)[0]
    # predictions = scaler.inverse_transform(predict(model, X).reshape(-1,1)).reshape(1,-1)[0]
    # predictions = model.predict(X).reshape(1,-1)[0]
    # y = scaler.inverse_transform(np.array(y).reshape(-1,1)).flatten()[:-1]
    y = scaler.inverse_transform(np.array(y).reshape(-1,1)).flatten()
    # y = np.array(y).reshape(1,-1)[0]
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
    kl.update_state(scale_data(y.reshape(-1,1)), scale_data(predictions.reshape(-1,1)))
    rmse = RootMeanSquaredError()
    rmse.update_state(scale_data(y.reshape(-1,1)), scale_data(predictions.reshape(-1,1)))
    mae = MeanAbsoluteError()
    mae.update_state(scale_data(y.reshape(-1,1)), scale_data(predictions.reshape(-1,1)))
    return df, mse(y, predictions), kl.result().numpy(), rmse.result().numpy(), mae.result().numpy()


def plot_history(history):
    # summarize history for root mean squared error
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('model rmse')
    plt.ylabel('root mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def store_results(hyperparams, history, results, full_name):
    df = pd.DataFrame(columns=['train time', 'layer number', 'input shape', 'output shape', 'layer size',
                               'hidden size', 'learning rate', 'epochs', 'mse train', 'kl train',
                               'rmse train', 'mae train', 'mse val', 'kl val', 'rmse val', 'mae val',
                               'mse test', 'kl test', 'rmse test', 'mae test'])
    list = hyperparams + history

    for col, i in zip(df, list):
        df[col] = [i]

    df = pd.concat([df, results], axis=1)
    # df.to_excel(f"../../data/results/RNN/{name[0]}{name[2]}_{name[1]}_trial 39.xlsx")
    if not os.path.isdir(f"../../data/results/RNN/{full_name}"):
        os.makedirs(f"../../data/results/RNN/{full_name}")
    df.to_excel(f"../../data/results/RNN/{full_name}/ 15 (36).xlsx")


def store_individual_losses(dict_individual_losses, full_name):
    df_iterative = pd.DataFrame.from_dict(dict_individual_losses)
    df_iterative["train mean"] = df_iterative["train loss"].mean()
    df_iterative["value mean"] = df_iterative["value loss"].mean()
    df_iterative["test mean"] = df_iterative["test loss"].mean()
    if not os.path.isdir(f"../../data/results/RNN/{full_name}"):
        os.makedirs(f"../../data/results/RNN/{full_name}")
    df_iterative.to_excel(f"../../data/results/RNN/{full_name}/losses 16 (36).xlsx")


def run_and_plot_predictions(model, x_train, y_train, x_val, y_val, x_test, y_test, scaler):

    res_train, mse_train, kl_train, rmse_train, mae_train = plot_predictions(model, x_train, y_train, scaler)
    print(mse_train, kl_train, rmse_train, mae_train)
    res_val, mse_val, kl_val, rmse_val, mae_val = plot_predictions(model, x_val, y_val, scaler)
    print(mse_val, kl_val, rmse_val, mae_val)
    res_test, mse_test, kl_test, rmse_test, mae_test = plot_predictions(model, x_test, y_test, scaler)
    print(mse_test, kl_test, rmse_test, mae_test)

    df_res = pd.concat([res_train, res_val, res_test], axis=1)
    result_values = [mse_train, kl_train, rmse_train, mae_train, mse_val, kl_val, rmse_val, mae_val,
                     mse_test, kl_test, rmse_test, mae_test]
    return df_res, result_values