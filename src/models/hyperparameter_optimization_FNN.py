import logging
import sys

import numpy as np
import optuna
import src.models.train_model as train
import yaml
from keras.layers import BatchNormalization, Dense, Dropout
from keras.models import Sequential
from optuna.integration.keras import KerasPruningCallback
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from src.features.build_features import (load_preprocessed_data,
                                         scale_event_data)
from tensorflow import keras

config = yaml.safe_load(open("src/config.yaml"))

data = load_preprocessed_data()
data["event_X_train"], data["event_X_test"] = scale_event_data(
    data["event_X_train"], data["event_X_test"]
)

METRICS = [
    keras.metrics.AUC(name="AUC"),
    train.f1_score,
]

# stops training early if score doesn't improve
early_stopping = keras.callbacks.EarlyStopping(
    monitor=config["RNN_params"]["monitor"],
    verbose=1,
    patience=5,
    mode=config["RNN_params"]["mode"],
    restore_best_weights=True,
)

class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(data["y_train"]), y=data["y_train"]
)
class_weights = {
    _class: weight for _class, weight in zip(np.unique(data["y_train"]), class_weights)
}

MONITOR = config["RNN_params"]["monitor"]
MODE = config["RNN_params"]["mode"]


def create_model(params: dict):
    """Generates a FNN model for hyperparameter training using Optuna

    Args:
        params (dict): contains Optuna-defined hyperparameter ranges

    Returns:
        keras model: a compiled model
    """

    ACTIVATION = config["RNN_params"]["activation"]

    model = Sequential()
    model.add(Dense(units=data["event_X_train"].shape[1], activation=ACTIVATION))

    for i in range(params["num_hidden_layers"]):
        model.add(BatchNormalization(epsilon=0.01))
        model.add(Dropout(params["dropout"]))
        model.add(Dense(units=params[f"n_units_1{i}"], activation=ACTIVATION))

    model.add(
        Dense(
            units=1,
            activation="sigmoid",
        )
    )

    OPTIMIZER = keras.optimizers.Adam(
        learning_rate=params["lr"],
        clipnorm=config["RNN_params"]["clipnorm"],
    )

    model.compile(optimizer=OPTIMIZER, loss="binary_crossentropy", metrics=METRICS)

    return model


def objective(trial):
    params = {
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
        "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 6),
        "lr": trial.suggest_loguniform("lr", 1e-4, 1e-1),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
    }

    for i in range(params["num_hidden_layers"]):
        params[f"n_units_1{i}"] = trial.suggest_int(f"n_units_l{i}", 20, 400)

    model = create_model(params)

    model.fit(
        data["event_X_train"],
        data["y_train"],
        batch_size=params["batch_size"],
        class_weight=class_weights,
        epochs=200,
        callbacks=[early_stopping, KerasPruningCallback(trial, "val_AUC")],
        validation_data=(data["event_X_test"], data["y_test"]),
        shuffle=True,
        verbose=1,
    )

    preds = model.predict(data["event_X_test"])
    auc_score = roc_auc_score(data["y_test"], preds)

    return auc_score


def main():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "bayesian_opt_FNN"  # Unique identifier of the study.
    storage_name = f"sqlite:///models/hyperparam_dbs/{study_name}.db"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_min_trials=5),
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=200, n_jobs=1)


if __name__ == "__main__":
    main()
