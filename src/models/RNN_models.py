import os

os.environ["PYTHONHASHSEED"] = str(1)  # sets python random seed for reproducibility

import tensorflow as tf
import wandb
from keras import Input, Model
from keras.layers import (
    LSTM,
    BatchNormalization,
    Concatenate,
    Dense,
    LayerNormalization,
)
from keras.models import Sequential
from sklearn.model_selection import KFold
from src.features.build_features import scale_event_data, scale_object_data
from tensorflow import keras


def get_average_history(histories):
    history = {}

    for i in histories:
        for k, v in i.items():
            if k not in history:
                history[k] = v
            else:
                history[k] += v

    history = {k: v / len(histories) for k, v in history.items()}
    return history


class NN_model:
    def __init__(self, dropout_type, loss, **kwargs):

        self.dropout_type = dropout_type
        self.loss = loss
        self.callbacks = set()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.optimizer = keras.optimizers.Adam(
            self.lr,
            clipnorm=self.clipnorm,
        )
        self.metrics = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="AUC"),
        ]

        if self.loss == "binary_crossentropy":
            self.batch_size = self.cross_entropy_batch_size
        else:
            self.batch_size = self.asimov_batch_size

    def save(self, model_filepath):
        # saves the network at regular intervals so you can pick the best version
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_filepath,
            monitor=self.monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode=self.mode,
            save_freq="epoch",
        )
        self.callbacks.add(checkpoint)

    def use_wandb(self):
        wandbcallback = wandb.keras.WandbCallback()
        self.callbacks.add(wandbcallback)

    def train(self, epochs, X_train, X_test, y_train, y_test, class_weights):
        # stops training early if score doesn't improve
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.monitor,
            verbose=1,
            patience=epochs // 2,
            mode=self.mode,
            restore_best_weights=True,
        )
        self.callbacks.add(early_stopping)

        X_train, X_test = self.scale_data(X_train=X_train, X_test=X_test)

        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            class_weight=class_weights,
            epochs=epochs,
            callbacks=list(self.callbacks),
            validation_data=(X_test, y_test),
            shuffle=True,
            verbose=1,
        )

        return history


class merged_model(NN_model):
    def __init__(self, dropout_type, loss, event_shape, object_shape, **kwargs):
        super().__init__(dropout_type, loss, **kwargs)
        self.event_shape = event_shape
        self.object_shape = object_shape
        self.make_model()

    def make_model(self):
        self.model = Sequential()
        self.model.add(
            LSTM(
                units=self.lstm_units,
                input_shape=self.object_shape,
                return_sequences=True,
                recurrent_dropout=self.redropout,
            )
        )
        self.model.add(LayerNormalization(axis=-1, center=True, scale=True))
        self.model.add(
            LSTM(
                units=self.lstm_units,
                recurrent_dropout=self.redropout,
            )
        )
        self.model.add(
            Dense(
                units=self.output_units,
                activation=self.activation,
            )
        )

        event_features = Input(shape=self.event_shape)
        merged_model = Concatenate()([event_features, self.model.output])

        for _ in range(self.num_merged_layers):
            merged_model = BatchNormalization(epsilon=0.01)(merged_model)
            merged_model = self.dropout_type(self.dropout)(merged_model)
            merged_model = Dense(units=self.merged_units, activation=self.activation)(
                merged_model
            )

        merged_model = Dense(1, activation="sigmoid")(merged_model)
        self.model = Model(
            inputs=[event_features, self.model.input], outputs=merged_model
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

    def scale_data(self, X_train: list, X_test):
        X_train[0], X_test[0] = scale_event_data(X_train[0], X_test[0])
        X_train[1], X_test[1] = scale_object_data(X_train[1], X_test[1])
        return (X_train, X_test)

    def cross_validate(self, epochs, X, y, class_weights, cv=1):
        histories = []
        kfold = KFold(n_splits=cv, shuffle=False)

        for train_idx, test_idx in kfold.split(X[0]):

            # split into train and test
            event_X_train, event_X_test = (X[0].iloc[train_idx], X[0].iloc[test_idx])
            object_X_train, object_X_test = (X[1][train_idx], X[1][test_idx])
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            X_train = [event_X_train, object_X_train]
            X_test = [event_X_test, object_X_test]

            # train model
            history = self.train(
                epochs, X_train, X_test, y_train, y_test, class_weights
            )
            histories.append(history)

        history = get_average_history(histories)
        return history


class RNN_model(NN_model):
    def __init__(self, dropout_type, loss, object_shape, **kwargs):
        super().__init__(dropout_type, loss, **kwargs)
        self.object_shape = object_shape
        self.make_model()

    def make_model(self):
        self.model = Sequential()
        self.model.add(
            LSTM(
                units=self.lstm_units,
                input_shape=self.object_shape,
                return_sequences=True,
                recurrent_dropout=self.redropout,
            )
        )
        self.model.add(LayerNormalization(axis=-1, center=True, scale=True))
        self.model.add(
            LSTM(
                units=self.lstm_units,
                recurrent_dropout=self.redropout,
            )
        )
        self.model.add(LayerNormalization(axis=-1, center=True, scale=True))
        self.model.add(
            Dense(
                units=1,
                activation="sigmoid",
            )
        )

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

    def scale_data(self, X_train, X_test):
        X_train, X_test = scale_event_data(X_train, X_test)

    def cross_validate(self, epochs, X, y, class_weights, cv=5):

        histories = []
        kfold = KFold(n_splits=cv, shuffle=False)

        for train_idx, test_idx in kfold.split(X):
            # split into train and test
            X_train, X_test = (X[train_idx], X[test_idx])
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # scale the data
            X_train, X_test = scale_object_data(X_train, X_test)

            # train model
            history = self.train(
                epochs, X_train, X_test, y_train, y_test, class_weights
            )
            histories.append(history)

        history = get_average_history(histories)
        return history
