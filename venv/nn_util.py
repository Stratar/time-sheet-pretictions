import tensorflow as tf


class LearningRateReducer(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if epoch > int((self.model.epochs) * (3/4)):
            old_lr = self.model.optimizer.lr.read_value()
            new_lr = old_lr/10
            print(f"\nThe learning rate has been reduced to: {new_lr}\n")
            self.model.optimizer.lr.assign(new_lr)