import argparse
import os

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import LSTM, BatchNormalization, Concatenate, Dense
from keras.models import Sequential
from sklearn.utils import class_weight
from src.features.build_features import load_preprocessed_data
from src.visualization.visualize import make_training_curves, save_plot
from tensorflow import keras


def make_RNN_model(data: dict):
    """Defines and compiles a recurrent neural network model

    Args:
        data (dict): data to be fed to the model

    Returns:
        model: A compiled RNN model
    """

    def f1_score(y_true, y_pred):  # taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    LR = 0.001
    ACTIVATION = "relu"
    num_layers = 2

    optimizer = keras.optimizers.Adam(
        learning_rate=LR,
        clipnorm=0.001,
    )
    metrics = [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="AUC"),
        f1_score,
    ]

    DNN_model = Input(shape=data["event_X_train"].shape[1])

    RNN_model = Sequential(
        [
            LSTM(
                200,
                input_shape=(
                    data["object_X_train"].shape[1],
                    data["object_X_train"].shape[2],
                ),
                activation="tanh",
                unroll=False,
            ),
            BatchNormalization(epsilon=0.01),
        ]
    )

    merged_model = Concatenate()([DNN_model, RNN_model.output])

    for _ in range(num_layers):
        merged_model = BatchNormalization(epsilon=0.01)(merged_model)
        merged_model = Dense(100, activation=ACTIVATION)(merged_model)

    merged_model = Dense(1, activation="sigmoid")(merged_model)

    model = Model(inputs=[DNN_model, RNN_model.input], outputs=merged_model)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)

    return model


def train_RNN(epochs: int, model_filepath: str, data: dict):
    BATCH_SIZE = 64
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(data["y_train"]), y=data["y_train"]
    )
    class_weights_dict = {
        _class: weight
        for _class, weight in zip(np.unique(data["y_train"]), class_weights)
    }

    MONITOR = "val_loss"
    MODE = "auto"

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

    model = make_RNN_model(data)

    history = model.fit(
        [data["event_X_train"], data["object_X_train"]],
        data["y_train"],
        batch_size=BATCH_SIZE,
        class_weight=class_weights_dict,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint],
        validation_data=([data["event_X_test"], data["object_X_test"]], data["y_test"]),
        shuffle=True,
        verbose=1,
    )

    return (history, model)


def main(args):
    EPOCHS = args.epochs
    MODEL_NAME = args.model_name
    MODEL_FILEPATH = os.path.join("models", MODEL_NAME)
    data = load_preprocessed_data(args.all_data)

    history, model = train_RNN(EPOCHS, MODEL_FILEPATH, data)
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
        type=bool,
        default=True,
        help="Whether to use all the background data",
    )

    args = parser.parse_args()
    main(args)
