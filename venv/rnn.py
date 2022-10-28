from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, KLDivergence, Accuracy, MeanAbsoluteError
from keras.optimizers import Adam
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # This is also a normalizer
import numpy as np
import time
from datetime import datetime


def lstm_model(in_shape, out_shape, n_layer_lim=2, lstm_size=128, hid_size=8):
    model = Sequential()
    model._name = "lstm"
    model.add(InputLayer(in_shape))
    n_layers = in_shape[0]-1
    if n_layers > n_layer_lim: n_layers = n_layer_lim
    for _ in range(n_layer_lim):
        model.add(Bidirectional(LSTM(lstm_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Bidirectional(LSTM(lstm_size, dropout=0.2, recurrent_dropout=0.2)))
    # model.add(Dense(hid_size, 'relu'))
    model.add(Dense(out_shape[0], 'sigmoid'))

    print(model.summary())

    cp = ModelCheckpoint('model_lstm/', save_best_only=True)
    return model, cp


'''Some things to note about the GRU networks: 
They can utilise the node dropout, allowing them to avoid overfitting from nodes in the input or recurrent layers.
The dropout parameter controls the fraction of units dropped for the linear transformation of the inputs.
The recurrent dropout is also a fraction of the units dropped for the linear transformation of the recurrent state.
The default activation of the GRU layer is tanh, which produces values in the range of -1 to 1.
The default recurrent activation is a sigmoid function, producing values between 0 and 1. 

'''


def gru_model(in_shape, out_shape, n_layer_lim=2, gru_size=128, hid_size=8): #gru_size=16 as well
    model = Sequential()
    model._name = "gru"
    model.add(InputLayer(in_shape))
    n_layers = in_shape[0]-1
    if n_layers > n_layer_lim: n_layers = n_layer_lim
    for _ in range(n_layers-1):
        model.add(GRU(gru_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(GRU(gru_size, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(hid_size, 'relu'))
    model.add(Dense(out_shape[0], 'sigmoid'))

    print(model.summary())

    cp = ModelCheckpoint('model_gru/', save_best_only=True)
    return model, cp


def conv_model(in_shape, out_shape, n_layer_lim=4, conv_size=145, hid_size=16):
    model = Sequential()
    model._name = "conv"
    model.add(InputLayer(in_shape))
    # Experimentally, it showed that having one less convolutional layer than the size of the input is best
    n_layers = in_shape[0]-1
    # initialiser = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None)
    if n_layers > n_layer_lim: n_layers = n_layer_lim
    for _ in range(n_layers):
        model.add(Conv1D(int(conv_size), kernel_size=3, kernel_initializer=None)) # kernel_size=2
        conv_size /= 2
    model.add(Flatten())
    model.add(Dense(hid_size, 'relu', kernel_initializer=None))
    model.add(Dense(out_shape[0], 'sigmoid'))

    print(model.summary())

    cp = ModelCheckpoint('model_conv/', save_best_only=True)
    return model, cp


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


def plot_predictions(model, X, y, scaler, start=0, end=150):

    predictions = scaler.inverse_transform(model.predict(X).reshape(-1,1)).reshape(1,-1)[0]
    # predictions = scaler.inverse_transform(predict(model, X).reshape(-1,1)).reshape(1,-1)[0]
    # predictions = model.predict(X).reshape(1,-1)[0]
    # y = scaler.inverse_transform(np.array(y).reshape(-1,1)).flatten()[:-1]
    y = scaler.inverse_transform(np.array(y).reshape(-1,1)).flatten()
    # y = np.array(y).reshape(1,-1)[0]
    df = res_dataframe(predictions, y)
    plt.plot(df['Predictions'][start:end])
    plt.plot(df['Actuals'][start:end])
    plt.title('Predictions vs Actuals')
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend(['Predictions', 'Actuals'], loc='upper left')
    plt.grid()
    # plt.show()
    kl = KLDivergence()
    kl.update_state(scale_data(y.reshape(-1,1)), scale_data(predictions.reshape(-1,1)))
    acc = Accuracy()
    acc.update_state(scale_data(y.reshape(-1,1)), scale_data(predictions.reshape(-1,1)))
    rmse = RootMeanSquaredError()
    rmse.update_state(scale_data(y.reshape(-1,1)), scale_data(predictions.reshape(-1,1)))
    mae = MeanAbsoluteError()
    mae.update_state(scale_data(y.reshape(-1,1)), scale_data(predictions.reshape(-1,1)))
    return df, mse(y, predictions), kl.result().numpy(), acc.result().numpy(), rmse.result().numpy(), mae.result().numpy()


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
    # #
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for KL-Divergence
    # plt.plot(history.history['kullback_leibler_divergence'])
    # plt.plot(history.history['val_kullback_leibler_divergence'])
    # plt.title('model KL-Divergence')
    # plt.ylabel('KL-Divergence')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()


def store_results(hyperparams, history, results, name):
    df = pd.DataFrame(columns=['train time', 'layer number', 'input shape', 'output shape', 'layer size',
                               'hidden size', 'learning rate', 'epochs', 'mse train', 'kl train', 'accuracy train',
                               'rmse train', 'mae train', 'mse val', 'kl val', 'accuracy val','rmse val', 'mae val',
                               'mse test', 'kl test', 'accuracy test', 'rmse test', 'mae test'])
    list = hyperparams + history

    for col, i in zip(df, list):
        df[col] = [i]

    df = pd.concat([df, results], axis=1)
    df.to_excel(f"../../data/results/RNN/{name[0]}{name[2]}_{name[1]}_trial 39.xlsx")
    # df.to_excel("../../data/results/RNN/demo 4.xlsx")


def run_rnn(x_train, y_train, x_val, y_val, split=-1, fit_range=1):
    n_layer_lim = 2
    layer_size = 30
    hid_size = 20
    lr = 5e-4
    epochs = 100

    x = lambda n : n.shape[1:] if(len(n.shape)>1) else [1]
    if split == 1:
        input_shape = x_train[0].shape[1:]
        output_shape = x(y_train[0])
    else:
        input_shape = x_train.shape[1:]
        output_shape = x(y_train)

    model1, cp1 = conv_model(input_shape, output_shape, n_layer_lim, layer_size, hid_size)
    model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=lr), metrics=[RootMeanSquaredError(), KLDivergence(), Accuracy()])

    early_stop = EarlyStopping(monitor='val_loss', patience=25)
    '''
    If I want to train the model on data by multiple flexworkers, differently than the way it is currently organised, 
    with one worker per database for example, then the same model can be trained, with the only thing needed being the
    model being fitted for every separate dataset. 
    '''
    start = datetime.now()
    if split == 1:
        for input, output, xval, yval in zip(x_train, y_train, x_val, y_val):
            history = model1.fit(x_train, y_train, shuffle=False, validation_data=[xval, yval], epochs=int(epochs/len(x_train)), callbacks=[cp1, early_stop])
    else:
        history = model1.fit(x_train, y_train, shuffle=False, validation_data=[x_val, y_val], epochs=epochs, callbacks=[cp1, early_stop])
    end = datetime.now()
    train_time = (end-start).total_seconds()
    print(f"The training time for the model was: {train_time}\n")
    plot_history(history)

    model1 = load_model('model_conv/')

    hyperparameters = [train_time, n_layer_lim, input_shape, output_shape, layer_size, hid_size, lr, epochs]
    return model1, hyperparameters
