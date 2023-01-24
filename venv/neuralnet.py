import tensorflow as tf
import numpy as np
from nn_util import LearningRateReducer
from tensorflow.keras import backend as K


def rmse(y_true, y_pred):
    return np.sqrt(tf.keras.losses.MeanSquaredError(float(y_true), float(y_pred)))


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = tf.keras.layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


class PositionEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.word_embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=output_dim
        )
        self.position_embedding_layer = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size,
            'output_dim': self.output_dim
        })
        return config

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices


class SemiTransformer:

    def __init__(       # in_shape, out_shape, n_layers=3, gru_size=10, hid_size=4, lr=1e-3
            self,
            input_shape,
            out_shape,
            num_transformer_blocks=2, # 5, 4
            head_size=256, # 256, 512
            num_heads=4, # 10, 4
            ff_dim=14, # 14, 4
            mlp_units=[128], #list of units per mlp layer | could be reduced to allow result jumps and spikes
            lr = 1e-4 # 1e-3
    ):
        mlp_dropout = 0.2
        dropout=0.2
        self.epochs = 1
        self.n_layers = num_transformer_blocks
        self.layer_size = head_size
        self.lr = lr
        self.hid_size = mlp_units[0]
        self.path = ''
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        # x = PositionEmbeddingLayer(input_shape[0], 250, 1)(x)
        # x = tf.squeeze(x, axis=3)
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = tf.keras.layers.Dense(dim, activation="relu")(x)
            x = tf.keras.layers.Dropout(mlp_dropout)(x)
        outputs = tf.keras.layers.Dense(out_shape[0], activation="relu")(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model._name = "Semi-Transformer"
        self.name = self.model._name
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=125)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_semitrans.h5', save_best_only=True)

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

    def set_path(self, path):
        self.path = path

    def set_learning_rate(self, lr):
        self.lr = lr
        K.set_value(self.model.optimizer.learning_rate, self.lr)

    def save(self, path):
        print("\nSaving model!\n")
        self.model.save_weights(path)

    def save_model(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load_weights(path)

    def disable_training(self):
        self.model.trainable = False

    def get_model(self):
        return self.model


class AdvGRUNeuralNetwork:

    def __init__(self, in_shape, out_shape, n_layers=3, gru_size=32, hid_size=4, lr=1e-4):
        self.n_layers = n_layers
        self.layer_size = gru_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1
        self.path = ''
        self.model = tf.keras.Sequential()
        self.model._name = "AdvGRU"
        self.name = self.model._name
        self.model.add(tf.keras.layers.InputLayer(in_shape, name='GRU_Input'))
        # self.model.add(tf.keras.layers.Dense(units=in_shape[1], activation='sigmoid', name='Dense_1'))
        for i in range(n_layers-1):
            self.model.add(tf.keras.layers.GRU(gru_size, dropout=0.2, recurrent_dropout=0, name=f'GRU_Hid_{i}'))
            self.model.add(tf.keras.layers.RepeatVector(out_shape[0], name=f'GRU_RepeatVector_{i}'))
        self.model.add(tf.keras.layers.GRU(gru_size, dropout=0.2, recurrent_dropout=0,
                                                                         return_sequences=True))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=out_shape[1], activation='sigmoid'), name='GRU_Output'))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150)
        self.cp = tf.keras.callbacks.ModelCheckpoint('model_advgru.h5', save_best_only=True) # Saving

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

    def set_path(self, path):
        self.path = path

    def check(self):
        return self.model.summary()

    def save(self, path):
        print("\nSaving model!\n")
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

    def __init__(self, in_shape, out_shape, path, n_layers=3, lstm_size=32, hid_size=4, ff_dim=4, lr=1e-4):
        self.n_layers = n_layers
        self.layer_size = lstm_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1

        self.path = ''

        # self.base_model = AdvLSTMNeuralNetwork(in_shape, out_shape, n_layers=n_layers, lstm_size=lstm_size)
        self.base_model = SemiTransformer(in_shape, out_shape, n_layers, lstm_size, hid_size, ff_dim)
        self.base_model.get_model().load_weights(path)
        '''
        Instead of adding the pre-made GRU network, make a new class that connects to the end of the generalised
        LSTM network.
        '''
        self.base_model.disable_training()

        self.model = tf.keras.models.Sequential()
        self.model.add(self.base_model.get_model())
        n_layers = int(n_layers)
        self.model.add(tf.keras.layers.RepeatVector(out_shape[0], name=f'Trans_RepeatVector'))
        # self.model.add(tf.keras.layers.Conv1D(lstm_size, 3, use_bias=False, activation='sigmoid'))
        for i in range(n_layers - 1):
        #     # self.model.add(tf.keras.layers.Dense(lstm_size, name='Trans_Dense'))
        #     self.model.add(tf.keras.layers.Conv1D(int(lstm_size/2), 2, use_bias=True, activation='sigmoid'))
        #     self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.GRU(int(lstm_size), dropout=0.2, recurrent_dropout=0))
            self.model.add(tf.keras.layers.RepeatVector(out_shape[0], name=f'Trans_RepeatVector_{i}'))

        # self.model.add(tf.keras.layers.Conv1D(int(lstm_size/3), 2, activation='sigmoid'))
        # self.model.add(tf.keras.layers.Flatten())
        # self.model.add(tf.keras.layers.RepeatVector(out_shape[0], name=f'Trans_RepeatVector'))
        self.model.add(tf.keras.layers.GRU(lstm_size, dropout=0.2, recurrent_dropout=0,
                                return_sequences=True, name='Trans_GRU'))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=out_shape[1], activation='sigmoid', name='Trans_output')))

        self.model._name = "TansferModel"
        self.name = self.model._name
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=125)
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

    def set_path(self, path):
        self.path = path

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        return self.model.summary()

    def save(self, path):
        print("\nSaving model!\n")
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

    def __init__(self, in_shape, out_shape, n_layers=3, lstm_size=64, hid_size=4, lr=1e-4):
        self.n_layers = n_layers
        self.layer_size = lstm_size
        self.hid_size = hid_size
        self.lr = lr
        self.epochs = 1
        self.full_name = ''
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
        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150)
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

    def set_full_name(self, full_name):
        self.full_name = full_name

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

