import argparse
import os

import keras.backend as K
import numpy as np
import tensorflow as tf
import yaml
from keras import Input, Model
from keras.layers import (
    LSTM,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    LayerNormalization,
)
from keras.models import Sequential
from sklearn.utils import class_weight
from src.features.build_features import load_preprocessed_data
from src.visualization.visualize import make_training_curves, save_plot
from tensorflow import keras

config = yaml.safe_load(open("src/config.yaml"))


class MonteCarloDropout(Dropout):
    """Keeps dropout on in testing mode for uncertainty predictions"""

    def call(self, inputs):
        return super().call(inputs, training=True)


def f1_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def make_RNN_model(data: dict, use_mc_dropout: bool = False):
    """Defines and compiles a recurrent neural network model

    Args:
        data (dict): data to be fed to the model
        use_mc_dropout (bool): whether or not to use dropout during testing

    Returns:
        model: A compiled RNN model
    """

    ACTIVATION = config["RNN_params"]["activation"]
    NUM_LAYERS = config["RNN_params"]["num_merged_layers"]
    DROPOUT = config["RNN_params"]["dropout"]
    REDROPOUT = config["RNN_params"]["redropout"]
    OPTIMIZER = keras.optimizers.Adam(
        learning_rate=config["RNN_params"]["lr"],
        clipnorm=config["RNN_params"]["clipnorm"],
    )
    METRICS = [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="AUC"),
        f1_score,
    ]

    DNN_model = Input(shape=data["event_X_train"].shape[1])

    RNN_model = Sequential()
    RNN_model.add(
        LSTM(
            units=config["RNN_params"]["lstm_units"],
            input_shape=(
                data["object_X_train"].shape[1],
                data["object_X_train"].shape[2],
            ),
            return_sequences=True,
            recurrent_dropout=REDROPOUT,
        )
    )
    RNN_model.add(LayerNormalization(axis=-1, center=True, scale=True))
    RNN_model.add(
        LSTM(
            units=config["RNN_params"]["lstm_units"],
            recurrent_dropout=REDROPOUT,
        )
    )
    RNN_model.add(LayerNormalization(axis=-1, center=True, scale=True))
    RNN_model.add(
        Dense(units=config["RNN_params"]["output_units"], activation=ACTIVATION)
    )
    RNN_model.add(BatchNormalization(epsilon=0.01))

    merged_model = Concatenate()([DNN_model, RNN_model.output])

    for _ in range(NUM_LAYERS):
        merged_model = BatchNormalization(epsilon=0.01)(merged_model)
        if use_mc_dropout:
            merged_model = MonteCarloDropout(DROPOUT)(merged_model)
        else:
            merged_model = Dropout(DROPOUT)(merged_model)
        merged_model = Dense(
            units=config["RNN_params"]["merged_units"], activation=ACTIVATION
        )(merged_model)

    merged_model = Dense(1, activation="sigmoid")(merged_model)

    model = Model(inputs=[DNN_model, RNN_model.input], outputs=merged_model)
    model.compile(optimizer=OPTIMIZER, loss="binary_crossentropy", metrics=METRICS)

    return model


def train_RNN(epochs: int, model_filepath: str, data: dict, model, save_model):
    BATCH_SIZE = config["RNN_params"]["batch_size"]
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(data["y_train"]), y=data["y_train"]
    )
    class_weights_dict = {
        _class: weight
        for _class, weight in zip(np.unique(data["y_train"]), class_weights)
    }

    MONITOR = config["RNN_params"]["monitor"]
    MODE = config["RNN_params"]["mode"]

    # stops training early if score doesn't improve
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=MONITOR,
        verbose=1,
        patience=epochs // 2,
        mode=MODE,
        restore_best_weights=True,
    )

    # saves the network at regular intervals so you can pick the best version
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_filepath,
        monitor=MONITOR,
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode=MODE,
        save_freq="epoch",
    )

    callbacks = [early_stopping]
    if save_model:
        callbacks.append(checkpoint)

    history = model.fit(
        [data["event_X_train"], data["object_X_train"]],
        data["y_train"],
        batch_size=BATCH_SIZE,
        class_weight=class_weights_dict,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=([data["event_X_test"], data["object_X_test"]], data["y_test"]),
        shuffle=True,
        verbose=1,
    )

    return history


def main(args):
    EPOCHS = args.epochs
    SAVE_MODEL = args.save
    MC_DROPOUT = args.mc_dropout
    MODEL_NAME = args.model_name
    MODEL_FILEPATH = os.path.join("models", MODEL_NAME)
    data = load_preprocessed_data(args.all_data)

    model = make_RNN_model(data, MC_DROPOUT)
    history = train_RNN(EPOCHS, MODEL_FILEPATH, data, model, SAVE_MODEL)
    if SAVE_MODEL:
        make_training_curves(history)
        save_plot(MODEL_NAME, "training_curves")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural net")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--model_name",
        default="model_test.h5",
        help="Name of model (ends in .h5)",
    )
    parser.add_argument(
        "--all_data",
        action="store_true",
        help="Use all of the available datasets",
    )
    parser.add_argument(
        "--mc_dropout",
        action="store_true",
        default=False,
        help="Add dropout during testing to calculate model uncertainty",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="save model to models folder and save create training curves",
    )

    args = parser.parse_args()
    main(args)
