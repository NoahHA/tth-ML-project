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
from src.models import train_model
from tensorflow import keras


class merged_model:
    def __init__(self, dropout_type, loss, event_shape, object_shape, **kwargs):

        self.dropout_type = dropout_type
        self.loss = loss
        self.event_shape = event_shape
        self.object_shape = object_shape
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
            train_model.f1_score,
        ]

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

    def use_wandb(self, data):
        wandbcallback = wandb.keras.WandbCallback(
            validation_data=(
                [data["event_X_test"], data["object_X_test"]],
                data["y_test"],
            ),
        )
        self.callbacks.add(wandbcallback)

    def train(self, epochs, data, class_weights):

        # stops training early if score doesn't improve
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.monitor,
            verbose=1,
            patience=epochs // 2,
            mode=self.mode,
            restore_best_weights=True,
        )
        self.callbacks.add(early_stopping)

        history = self.model.fit(
            [data["event_X_train"], data["object_X_train"]],
            data["y_train"],
            batch_size=self.batch_size,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=list(self.callbacks),
            validation_data=(
                [data["event_X_test"], data["object_X_test"]],
                data["y_test"],
            ),
            shuffle=True,
            verbose=1,
        )

        return history
