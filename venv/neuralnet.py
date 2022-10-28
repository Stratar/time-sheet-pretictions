import tensorflow as tf


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

    def check(self):
        return self.model.summary()
