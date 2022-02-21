import logging
import os
import sys

import keras.backend as K
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
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
from optuna.integration.keras import KerasPruningCallback
from sklearn.utils import class_weight
from src.features.build_features import preprocess_data
from tensorflow import keras

# LOADING PREPROCESSED DATA

load_path = r"data/processed"

use_all_data = input("Use all data (y/n): ")
if use_all_data != "y":
    preprocess_data()

event_X_train = pd.read_pickle(os.path.join(load_path, "event_X_train.pkl"))
event_X_test = pd.read_pickle(os.path.join(load_path, "event_X_test.pkl"))

y_train = pd.read_pickle(os.path.join(load_path, "y_train.pkl"))
y_test = pd.read_pickle(os.path.join(load_path, "y_test.pkl"))

object_X_train = np.load(os.path.join(load_path, "object_X_train.npy"))
object_X_test = np.load(os.path.join(load_path, "object_X_test.npy"))

ACTIVATION = "relu"


def f1_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


METRICS = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.AUC(name="AUC"),
    f1_score,
]

# stops training early if score doesn't improve
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    verbose=1,
    patience=3,
    mode="auto",
    restore_best_weights=True,
)

class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights = {
    _class: weight for _class, weight in zip(np.unique(y_train), class_weights)
}

MONITOR = "val_loss"
MODE = "auto"

# CREATING AND TRAINING MODEL


def create_model(params: dict):
    """Generates a merged ANN and RNN model for hyperparameter training using Optuna


    Args:
        params (dict): contains Optuna-defined hyperparameter ranges

    Returns:
        keras model: a compiled model
    """
    DNN_model = Input(shape=event_X_train.shape[1])

    RNN_model = Sequential()

    if params["lstm_layer_2"]:
        RNN_model.add(
            LSTM(
                params["lstm_units"],
                input_shape=(object_X_train.shape[1], object_X_train.shape[2]),
                activation="tanh",
                return_sequences=True,
                recurrent_dropout=params["redropout"],
            )
        )
        RNN_model.add(LayerNormalization(axis=-1, center=True, scale=True))

        RNN_model.add(
            LSTM(
                params["lstm_units"],
                activation="tanh",
                recurrent_dropout=params["redropout"],
            )
        )
        RNN_model.add(LayerNormalization(axis=-1, center=True, scale=True))

    else:
        RNN_model.add(
            LSTM(
                params["lstm_units"],
                input_shape=(object_X_train.shape[1], object_X_train.shape[2]),
                activation="tanh",
                recurrent_dropout=params["redropout"],
            )
        )
        RNN_model.add(LayerNormalization(axis=-1, center=True, scale=True))

    merged_model = Concatenate()([DNN_model, RNN_model.output])

    for _ in range(params["num_merged_layers"]):
        merged_model = BatchNormalization(epsilon=0.01)(merged_model)
        merged_model = Dropout(params["dropout"])(merged_model)
        merged_model = Dense(params["merged_units"], activation=ACTIVATION)(
            merged_model
        )

    merged_model = Dense(1, activation="sigmoid")(merged_model)

    OPTIMIZER = keras.optimizers.Adam(
        learning_rate=params["lr"],
        clipnorm=0.001,
    )

    model = Model(inputs=[DNN_model, RNN_model.input], outputs=merged_model)
    model.compile(optimizer=OPTIMIZER, loss="binary_crossentropy", metrics=METRICS)

    return model


def objective(trial):
    params = {
        "lstm_units": trial.suggest_int("lstm_units", 40, 400, 20),
        "merged_units": trial.suggest_int("merged_units", 40, 400, 20),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
        "redropout": trial.suggest_uniform("redropout", 0.0, 0.5),
        "num_merged_layers": trial.suggest_int("num_merged_layers", 1, 4),
        "lstm_layer_2": trial.suggest_categorical("lstm_layer2", [True, False]),
        "lr": trial.suggest_loguniform("lr", 1e-4, 1e-1),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }

    model = create_model(params)

    model.fit(
        [event_X_train, object_X_train],
        y_train,
        batch_size=params["batch_size"],
        class_weight=class_weights,
        epochs=100,
        callbacks=[early_stopping, KerasPruningCallback(trial, "val_loss")],
        validation_data=([event_X_test, object_X_test], y_test),
        shuffle=True,
        verbose=1,
    )

    score = model.evaluate([event_X_test, object_X_test], y_test)
    return -score[0]


def main():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "bayesian_opt_v2"  # Unique identifier of the study.
    storage_name = f"sqlite:///models/{study_name}.db"

    # optuna.delete_study(study_name=study_name, storage=storage_name)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_min_trials=5),
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=200, n_jobs=-1, show_progress_bar=True)


if __name__ == "__main__":
    main()
