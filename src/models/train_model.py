import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, LSTM, Concatenate, BatchNormalization
from keras import Model
import pandas as pd
import sys
from src.features.build_features import preprocess_data

# command-line argument is the name under which the model will be saved

############## LOADING PREPROCESSED DATA ##############

load_path = r"data/processed"
preprocess_data()

event_X_train = pd.read_pickle(os.path.join(load_path, "event_X_train.pkl"))
event_X_test = pd.read_pickle(os.path.join(load_path, "event_X_test.pkl"))

y_train = pd.read_pickle(os.path.join(load_path, "y_train.pkl"))
y_test = pd.read_pickle(os.path.join(load_path, "y_test.pkl"))

object_X_train = np.load(os.path.join(load_path, "object_X_train.npy"))
object_X_test = np.load(os.path.join(load_path, "object_X_test.npy"))

############## DEFINING HYPERPARAMETERS ##############

LR = 0.001
ACTIVATION = "relu"
BATCH_SIZE = 64
EPOCHS = int(input("Number of epochs: "))
model_name = sys.argv[1]
MODEL_FILEPATH = os.path.join("models", model_name)

OPTIMIZER = keras.optimizers.Adam(
    learning_rate=LR,
)


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

class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights = {l: c for l, c in zip(np.unique(y_train), class_weights)}

MONITOR = "val_loss"
MODE = "auto"

# stops training early if score doesn't improve
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=MONITOR,
    verbose=1,
    patience=20,
    mode=MODE,
    restore_best_weights=True,
)

# saves the network at regular intervals so you can pick the best version
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_FILEPATH,
    monitor=MONITOR,
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode=MODE,
    save_freq="epoch",
)

# reduces the lr whenever training plateaus
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor=MONITOR,
    factor=0.1,
    patience=3,
    mode=MODE,
)

############## CREATING AND TRAINING MODEL ##############


def create_model():
    DNN_model = Sequential(
        [
            Dense(40, input_shape=(event_X_train.shape[1],), activation=ACTIVATION),
            BatchNormalization(),
        ]
    )

    RNN_model = Sequential(
        [
            LSTM(
                200,
                input_shape=(object_X_train.shape[1], object_X_train.shape[2]),
                activation="tanh",
                unroll=False,
            ),
            BatchNormalization(),
        ]
    )

    merged_model = Concatenate()([DNN_model.output, RNN_model.output])

    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(40, activation=ACTIVATION)(merged_model)
    merged_model = Dense(1, activation="sigmoid")(merged_model)

    model = Model(inputs=[DNN_model.input, RNN_model.input], outputs=merged_model)
    model.compile(optimizer=OPTIMIZER, loss="binary_crossentropy", metrics=METRICS)

    return model


def make_training_curves(history):
    # saves training curves

    plot_path = r"reports/figures"
    plot_path = os.path.join(plot_path, model_name)

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    metrics = ["loss", "accuracy", "AUC", "f1_score"]
    fig = plt.figure(figsize=(20, 10))

    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ")
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label="Train")
        plt.plot(
            history.epoch, history.history["val_" + metric], linestyle="--", label="Val"
        )

        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.legend()

    plt.savefig(os.path.join(plot_path, "training_curves.png"))


def main():
    model = create_model()

    history = model.fit(
        [event_X_train, object_X_train],
        y_train,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint],
        validation_data=([event_X_test, object_X_test], y_test),
        shuffle=True,
        verbose=1,
    )

    make_training_curves(history)


if __name__ == "__main__":
    main()
