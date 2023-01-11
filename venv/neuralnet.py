import tensorflow as tf
import numpy as np
from nn_util import LearningRateReducer


def rmse(y_true, y_pred):
    return np.sqrt(tf.keras.losses.MeanSquaredError(float(y_true), float(y_pred)))



class Embedder:
    def __init__(self, unique_cat_size, embedding_size, n_layers, gru_size, learning_rate, in_shape, out_shape):
        categorical_embedding_model = tf.keras.Sequential()
        input_categorical_data = tf.keras.layers.Input(shape=(unique_cat_size,))
        categorical_embedding_model.add(input_categorical_data)
        embedder = tf.keras.layers.Embedding(input_dim=unique_cat_size, output_dim=embedding_size,
                                             embeddings_initializer=tf.keras.initializers.RandomUniform(
                                                 seed=42))#Give some constant seed for the replicability of the results
        embedder.trainable = False  # Disable training in order to get consistent vector representations
        categorical_embedding_model.add(embedder)
        # embedder_flat = tf.keras.layers.Flatten(embedder)
        embedder_flat = tf.keras.layers.Flatten()
        categorical_embedding_model.add(embedder_flat)

        self.n_layers = n_layers
        self.layer_size = gru_size
        self.learning_rate = learning_rate
        self.epochs = 1
        self.model = tf.keras.Sequential()
        self.model._name = "AdvGRU"
        self.name = self.model._name
        self.model.add(tf.keras.layers.InputLayer(in_shape))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_size, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(tf.keras.layers.RepeatVector(out_shape[0]))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_size, dropout=0.2, recurrent_dropout=0.2,
                                                                         return_sequences=True)))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=out_shape[1], activation='sigmoid')))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_advgru/', save_best_only=True)


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


class AdvGRUNeuralNetwork:

    def __init__(self, in_shape, out_shape, n_layers=3, gru_size=32, hid_size=4, lr=2.5e-4):
        self.n_layers = n_layers
        self.layer_size = gru_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1
        self.model = tf.keras.Sequential()
        self.model._name = "AdvGRU"
        self.name = self.model._name
        self.model.add(tf.keras.layers.InputLayer(in_shape, name='GRU_Input'))
        for i in range(n_layers-1):
            self.model.add(tf.keras.layers.GRU(gru_size, dropout=0.3, recurrent_dropout=0))
            self.model.add(tf.keras.layers.RepeatVector(out_shape[0], name=f'GRU_RepeatVector_{i}'))
        self.model.add(tf.keras.layers.GRU(gru_size, dropout=0.3, recurrent_dropout=0,
                                                                         return_sequences=True))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=out_shape[1], activation='sigmoid'), name='GRU_Output'))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=550)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_advgru.h5', save_best_only=False) # Saving

    def compile(self):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                                                        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                                        tf.keras.metrics.KLDivergence()])

    def fit(self, x, y, x_val, y_val, epochs=100):
        self.epochs = epochs
        history = self.model.fit(x, y, shuffle=False, validation_data=[x_val, y_val], epochs=epochs,
                                 callbacks=[self.cp, self.early_stop]) # Remove the LR reducer if need be
        return history

    def evaluate(self, x, y):
        _, acc = self.model.evaluate(x, y)
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        return self.model.summary()

    def save(self, path):
        self.model.save_weights(path)

    def save_model(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load_weights(path)

    def disable_training(self):
        self.model.trainable = False

    def get_model(self):
        return self.model


class TransferNeuralNetwork:

    def __init__(self, in_shape, out_shape, path, n_layers=3, lstm_size=32, hid_size=4, lr=1e-4):
        self.n_layers = n_layers
        self.layer_size = lstm_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1

        self.base_model = AdvLSTMNeuralNetwork(in_shape, out_shape, n_layers=n_layers, lstm_size=lstm_size)
        self.base_model.get_model().load_weights(path)
        '''
        Instead of adding the pre-made GRU network, make a new class that connects to the end of the generalised
        LSTM network.
        '''
        self.base_model.disable_training()

        self.model = tf.keras.models.Sequential()
        self.model.add(self.base_model.get_model())
        n_layers = int(n_layers)
        # self.model.add(tf.keras.layers.Conv1D(lstm_size, 3, use_bias=False, activation='sigmoid'))
        for i in range(n_layers - 1):
        #     # self.model.add(tf.keras.layers.Dense(lstm_size, name='Trans_Dense'))
        #     self.model.add(tf.keras.layers.Conv1D(int(lstm_size/2), 2, use_bias=True, activation='sigmoid'))
        #     self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.GRU(int(lstm_size), dropout=0.2, recurrent_dropout=0))
            self.model.add(tf.keras.layers.RepeatVector(out_shape[0], name=f'Trans_RepeatVector_{i}'))

        self.model.add(tf.keras.layers.Conv1D(int(lstm_size/3), 2, activation='sigmoid'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.RepeatVector(out_shape[0], name=f'Trans_RepeatVector'))
        self.model.add(tf.keras.layers.GRU(lstm_size, dropout=0.2, recurrent_dropout=0,
                                return_sequences=True, name='Trans_GRU'))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=out_shape[1], activation='sigmoid', name='Trans_output')))

        self.model._name = "TansferModel"
        self.name = self.model._name
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_advgru.h5', save_best_only=True)

    def compile(self):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                                                        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                                        tf.keras.metrics.KLDivergence()])

    def build(self, in_win_size=14, n_vars=6):
        # Pass the variables into the build itself
        self.model.build(input_shape=(None, in_win_size, n_vars))

    def fit(self, x, y, x_val, y_val, epochs=100):
        self.epochs = epochs
        history = self.model.fit(x, y, shuffle=False, validation_data=[x_val, y_val], epochs=epochs,
                                 callbacks=[self.cp, self.early_stop]) # Remove the LR reducer if need be
        return history

    def evaluate(self, x, y):
        _, acc = self.model.evaluate(x, y)
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        return self.model.summary()

    def save(self, path):
        self.model.save_weights(path)

    def save_model(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load_weights(path)

    def disable_training(self):
        self.model.trainable = False

    def get_model(self):
        return self.model


class AdvLSTMNeuralNetwork:

    def __init__(self, in_shape, out_shape, n_layers=3, lstm_size=64, hid_size=4, lr=5e-4):
        self.n_layers = n_layers
        self.layer_size = lstm_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1
        self.model = tf.keras.Sequential()
        self.model._name = "AdvLSTM"
        self.name = self.model._name
        self.model.add(tf.keras.layers.InputLayer(in_shape))
        for i in range(n_layers-1):
            self.model.add(tf.keras.layers.LSTM(lstm_size, dropout=0.2, recurrent_dropout=0))
            self.model.add(tf.keras.layers.RepeatVector(out_shape[0]))
        self.model.add(tf.keras.layers.LSTM(lstm_size, dropout=0.2, recurrent_dropout=0,
                                                                         return_sequences=True))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=out_shape[1], activation='sigmoid')))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # val_loss cannot be found in early stopping condition, it only detects the metrics
        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=75)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_advlstm.h5', save_best_only=False)

    def compile(self):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                                                        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                                        tf.keras.metrics.KLDivergence()])

    def fit(self, x, y, x_val, y_val, epochs=100):
        self.epochs = epochs
        history = self.model.fit(x, y, shuffle=False, validation_data=[x_val, y_val], epochs=epochs,
                                 callbacks=[self.cp, self.early_stop]) # Remove the LR reducer if need be
        return history

    def evaluate(self, x, y):
        _, acc = self.model.evaluate(x, y)
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        return self.model.summary()

    def save(self, path):
        self.model.save_weights(path)

    def save_model(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load_weights(path)

    def disable_training(self):
        self.model.trainable = False

    def get_model(self):
        return self.model

