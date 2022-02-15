import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import keras_tuner as kt
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import (
    Dense,
    LSTM,
    Concatenate,
    BatchNormalization,
    LayerNormalization,
    Dropout,
)
from keras import Model
import pandas as pd


############## LOADING PREPROCESSED DATA ##############

load_path = r"data/processed"

event_X_train = pd.read_pickle(os.path.join(load_path, "event_X_train.pkl"))
event_X_test = pd.read_pickle(os.path.join(load_path, "event_X_test.pkl"))

y_train = pd.read_pickle(os.path.join(load_path, "y_train.pkl"))
y_test = pd.read_pickle(os.path.join(load_path, "y_test.pkl"))

object_X_train = np.load(os.path.join(load_path, "object_X_train.npy"))
object_X_test = np.load(os.path.join(load_path, "object_X_test.npy"))

############## DEFINING HYPERPARAMETERS ##############

ACTIVATION = "relu"
BATCH_SIZE = 64


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
    patience=2,
    mode="auto",
    restore_best_weights=True,
)

class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights = {l: c for l, c in zip(np.unique(y_train), class_weights)}

MONITOR = "val_loss"
MODE = "auto"

############## CREATING AND TRAINING MODEL ##############


def create_model(hp):
    units1 = hp.Int("dnn_units", min_value=40, max_value=300, step=20)
    units2 = hp.Int("lstm_units", min_value=40, max_value=300, step=20)
    units3 = hp.Int("merged_units", min_value=40, max_value=300, step=20)
    dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05)
    re_dropout = hp.Float("redroput", min_value=0.0, max_value=0.5, step=0.05)
    merged_layer_2 = hp.Boolean("layer4")
    merged_layer_3 = hp.Boolean("layer5")
    lstm_layer_2 = hp.Boolean("lstm_layer_2")
    lr = hp.Choice("lr", [1e-4, 1e-3, 1e-2])

    DNN_model = Sequential()
    DNN_model.add(
        Dense(units1, input_shape=(event_X_train.shape[1],), activation=ACTIVATION)
    )
    DNN_model.add(BatchNormalization())
    DNN_model.add(Dropout(dropout))

    RNN_model = Sequential()

    if lstm_layer_2:
        RNN_model.add(
            LSTM(
                units2,
                input_shape=(object_X_train.shape[1], object_X_train.shape[2]),
                activation="tanh",
                return_sequences=True,
                recurrent_dropout=re_dropout,
            )
        )
        RNN_model.add(LayerNormalization(axis=-1, center=True, scale=True))

        RNN_model.add(
            LSTM(
                units2,
                input_shape=(object_X_train.shape[1], object_X_train.shape[2]),
                activation="tanh",
                recurrent_dropout=re_dropout,
            )
        )
        RNN_model.add(LayerNormalization(axis=-1, center=True, scale=True))

    else:
        RNN_model.add(
            LSTM(
                units2,
                input_shape=(object_X_train.shape[1], object_X_train.shape[2]),
                activation="tanh",
                recurrent_dropout=re_dropout,
            )
        )
        RNN_model.add(LayerNormalization(axis=-1, center=True, scale=True))

    merged_model = Concatenate()([DNN_model.output, RNN_model.output])
    merged_model = Dense(units3, activation=ACTIVATION)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dropout(dropout)(merged_model)

    if merged_layer_2:
        merged_model = Dense(units3, activation=ACTIVATION)(merged_model)
        merged_model = BatchNormalization()(merged_model)
        merged_model = Dropout(dropout)(merged_model)

        if merged_layer_3:
            merged_model = Dense(units3, activation=ACTIVATION)(merged_model)
            merged_model = BatchNormalization()(merged_model)
            merged_model = Dropout(dropout)(merged_model)

    merged_model = Dense(1, activation="sigmoid")(merged_model)
    model = Model(inputs=[DNN_model.input, RNN_model.input], outputs=merged_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=METRICS,
    )

    return model


def main():
    bayesian_opt_tuner = kt.BayesianOptimization(
        create_model,
        objective="val_loss",
        max_trials=200,
        executions_per_trial=1,
        directory="models/tmp/tb",
        project_name="binary_model_all_data",
        overwrite=True,
    )

    bayesian_opt_tuner.search(
        [event_X_train, object_X_train],
        y_train,
        epochs=1, #100
        batch_size=BATCH_SIZE,
        validation_data=([event_X_test, object_X_test], y_test),
        callbacks=[early_stopping],
        class_weight=class_weights,
    )


if __name__ == "__main__":
    main()
