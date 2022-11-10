import tensorflow as tf
import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(tf.keras.losses.MeanSquaredError(float(y_true), float(y_pred)))


class NeuralNet:

    def __init__(self, in_shape, out_size=5, n_layers=2, size_layer=300, lr=0.01):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(500, input_shape=in_shape, activation='relu'))
        for _ in range(n_layers):
            self.model.add(tf.keras.layers.Dense(size_layer, activation='relu'))
        self.model.add(tf.keras.layers.Dense(out_size, activation='sigmoid'))
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    def compile(self):
        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])

    def fit(self, x, y, epochs=100, batch_size=100):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)

    def evaluate(self, x, y):
        _, acc = self.model.evaluate(x, y)
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        return self.model.summary()


class ConvolutionalNeuralNetwork:

    def __init__(self, in_shape, out_shape, n_layers=2, conv_size=32, hid_size=8, lr=1e-3):
        self.n_layers = n_layers
        self.layer_size = conv_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1
        self.model = tf.keras.Sequential()
        self.model._name = "convolution"
        self.name = self.model._name
        self.model.add(tf.keras.layers.InputLayer(in_shape))
        for _ in range(n_layers):
            self.model.add(tf.keras.layers.Conv1D(int(conv_size), kernel_size=2, kernel_initializer=None)) # kernel_size=2
            conv_size /= 2
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(hid_size, 'relu', kernel_initializer=None))
        self.model.add(tf.keras.layers.Dense(out_shape[0], 'sigmoid'))
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_conv/', save_best_only=True)

    def compile(self):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                                                        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                                        tf.keras.metrics.KLDivergence(), tf.keras.metrics.Accuracy()])

    def fit(self, x, y, x_val, y_val, epochs=100, batch_size=52):
        self.epochs = epochs
        history = self.model.fit(x, y, shuffle=False, validation_data=[x_val, y_val], epochs=epochs,
                                 batch_size=batch_size, callbacks=[self.cp, self.early_stop])
        return history

    def evaluate(self, x, y):
        _, acc = self.model.evaluate(x, y)
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        return self.model.summary()


class GRUNeuralNetwork:

    def __init__(self, in_shape, out_shape, n_layers=3, gru_size=10, hid_size=4, lr=1e-3):
        self.n_layers = n_layers
        self.layer_size = gru_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1
        self.model = tf.keras.Sequential()
        self.model._name = "GRU"
        self.name = self.model._name
        self.model.add(tf.keras.layers.InputLayer(in_shape))
        for _ in range(n_layers-1):
            self.model.add(tf.keras.layers.GRU(gru_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(tf.keras.layers.GRU(gru_size, dropout=0.2, recurrent_dropout=0.2))
        # self.model.add(tf.keras.layers.Dense(hid_size, 'relu', kernel_initializer=None))
        self.model.add(tf.keras.layers.Dense(out_shape[0], 'sigmoid'))
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_gru/', save_best_only=True)

    def compile(self):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                                                        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                                        tf.keras.metrics.KLDivergence(), tf.keras.metrics.Accuracy()])

    def fit(self, x, y, x_val, y_val, epochs=100):
        self.epochs = epochs
        history = self.model.fit(x, y, shuffle=False, validation_data=[x_val, y_val], epochs=epochs,
                                 callbacks=[self.cp, self.early_stop])
        return history

    def evaluate(self, x, y):
        _, acc = self.model.evaluate(x, y)
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        return self.model.summary()


class GRUEmbeddedNeuralNetwork:

    def __init__(self, in_shape, out_shape, n_layers=2, gru_size=32, hid_size=16, lr=5e-5):
        self.n_layers = n_layers
        self.layer_size = gru_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1
        self.model = tf.keras.Sequential()
        self.model._name = "GRU"
        self.name = self.model._name
        # Insert the embedding input before the LSTM receives the general input.
        cat_input = tf.keras.layers.InputLayer()
        id_input = tf.keras.layers.Input()
        num_input = tf.keras.layers.InputLayer() # Restricted to only the dimensions of the numeric input data
        em = tf.keras.layers.Embedding()(cat_input) # To receive the ohe input for categorical data, or ids
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_size, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(tf.keras.layers.RepeatVector(out_shape[0]))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_size, dropout=0.2, recurrent_dropout=0.2,
                                                                         return_sequences=True)))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=out_shape[1], activation='sigmoid')))
        self.model.add(tf.keras.layers.Dense(out_shape[1], 'sigmoid'))
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_gru/', save_best_only=True)

    def compile(self):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                                                        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                                        tf.keras.metrics.KLDivergence(), tf.keras.metrics.Accuracy()])

    def fit(self, x, y, x_val, y_val, epochs=100):
        self.epochs = epochs
        history = self.model.fit(x, y, shuffle=False, validation_data=[x_val, y_val], epochs=epochs,
                                 callbacks=[self.cp, self.early_stop])
        return history

    def evaluate(self, x, y):
        _, acc = self.model.evaluate(x, y)
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        return self.model.summary()


class AdvGRUNeuralNetwork:

    def __init__(self, in_shape, out_shape, n_layers=3, gru_size=64, hid_size=4, lr=1e-4):
        self.n_layers = n_layers
        self.layer_size = gru_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1
        self.model = tf.keras.Sequential()
        self.model._name = "AdvGRU"
        self.name = self.model._name
        self.model.add(tf.keras.layers.InputLayer(in_shape))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_size, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(tf.keras.layers.RepeatVector(out_shape[0]))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_size, dropout=0.2, recurrent_dropout=0.2,
                                                                         return_sequences=True)))
        # self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=out_shape[1], activation='sigmoid')))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=out_shape[1], activation='sigmoid')))
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_advgru/', save_best_only=True)

    def compile(self):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                                                        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                                        tf.keras.metrics.KLDivergence(), tf.keras.metrics.Accuracy()])

    def fit(self, x, y, x_val, y_val, epochs=100):
        self.epochs = epochs
        history = self.model.fit(x, y, shuffle=False, validation_data=[x_val, y_val], epochs=epochs,
                                 callbacks=[self.cp, self.early_stop])
        return history

    def evaluate(self, x, y):
        _, acc = self.model.evaluate(x, y)
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        return self.model.summary()


class AdvLSTMNeuralNetwork:

    def __init__(self, in_shape, out_shape, n_layers=3, lstm_size=32, hid_size=4, lr=1e-4):
        self.n_layers = n_layers
        self.layer_size = lstm_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1
        self.model = tf.keras.Sequential()
        self.model._name = "AdvLSTM"
        self.name = self.model._name
        self.model.add(tf.keras.layers.InputLayer(in_shape))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, dropout=0.1, recurrent_dropout=0.1)))
        self.model.add(tf.keras.layers.RepeatVector(out_shape[0]))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, dropout=0.1, recurrent_dropout=0.1,
                                                                         return_sequences=True)))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=out_shape[1], activation='sigmoid')))
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_advlstm/', save_best_only=False)

    def compile(self):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                                                        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                                        tf.keras.metrics.KLDivergence(), tf.keras.metrics.Accuracy()])

    def fit(self, x, y, x_val, y_val, epochs=100):
        self.epochs = epochs
        history = self.model.fit(x, y, shuffle=False, validation_data=[x_val, y_val], epochs=epochs,
                                 callbacks=[self.cp, self.early_stop])
        return history

    def evaluate(self, x, y):
        _, acc = self.model.evaluate(x, y)
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        return self.model.summary()


class ConvLSTMNeuralNetwork:

    def __init__(self, in_shape, out_shape, n_layers=3, lstm_size=32, hid_size=4, lr=1e-4):
        self.n_layers = n_layers
        self.layer_size = lstm_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1
        self.model = tf.keras.Sequential()
        self.model._name = "ConvLSTM"
        self.name = self.model._name
        '''
        The input dimensions are incopatible for the conv1D, it expects 4, it receives 3
        THIS WON'T WORK 
        '''
        self.model.add(tf.keras.layers.InputLayer(in_shape))
        self.model.add(tf.keras.layers.ConvLSTM1D(lstm_size, kernel_size=2, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(tf.keras.layers.RepeatVector(out_shape[0]))
        self.model.add(tf.keras.layers.ConvLSTM1D(lstm_size, kernel_size=2, dropout=0.2, recurrent_dropout=0.2,
                                                                         return_sequences=True))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=out_shape[1], activation='sigmoid')))
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_convlstm/', save_best_only=True)

    def compile(self):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                                                        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                                        tf.keras.metrics.KLDivergence(), tf.keras.metrics.Accuracy()])

    def fit(self, x, y, x_val, y_val, epochs=100):
        self.epochs = epochs
        history = self.model.fit(x, y, shuffle=False, validation_data=[x_val, y_val], epochs=epochs,
                                 callbacks=[self.cp, self.early_stop])
        return history

    def evaluate(self, x, y):
        _, acc = self.model.evaluate(x, y)
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        return self.model.summary()