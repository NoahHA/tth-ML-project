"""
Train a machine learning model for ttH classification

Typical usage: train_model.py --epochs {epochs} --all_data -m RNN --wandb

where:
--epochs epochs: number of epochs to run training for (typically 5–50)
--model_type: which type of model to train, choice between RNN, FNN, merged RNN+FNN
                or multiclass RNN+FNN
--all_data: if added then model will automatically use all backgrounds,
                otherwise you have to specify
--model_type: which type of model to train, choice between RNN, FNN, merged, multiclass
--wandb: flag to save model and config to wanbd
--asimov_loss: flag to use asimov significance as the loss function,
                rather than cross-entropy
--mc_dropout: flag to use monte carlo dropout for uncertainty estimation
                (this will cause dropout to still be active during testing)
-h: show help message
"""

import argparse
import os
import random
from pathlib import Path

os.environ["PYTHONHASHSEED"] = str(1)  # sets python random seed for reproducibility

import keras.backend as K
import numpy as np
import tensorflow as tf
import wandb
import yaml
from keras.layers import LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from src.features import build_features
from src.models import models, significance_loss

config = yaml.safe_load(open(os.path.join(Path(__file__).parent.parent, "config.yaml")))


class MonteCarloDropout(Dropout):
    """Keeps dropout on in testing mode for uncertainty predictions"""

    def call(self, inputs):
        return super().call(inputs, training=True)


class MCLSTM(LSTM):
    def __init__(self, units, **kwargs):
        super(MCLSTM, self).__init__(units, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(MCLSTM, self).call(
            inputs,
            mask=mask,
            training=True,
            initial_state=initial_state,
        )


def reset_random_seeds():
    """Makes the experiment reproducible"""
    os.environ["PYTHONHASHSEED"] = str(1)
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)


def f1_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def calculate_class_weights(y_train):
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = {
        _class: weight for _class, weight in zip(np.unique(y_train), class_weights)
    }

    return class_weights_dict


def asimov_loss(y_train):
    signal_frac = sum(y_train == 1) / sum(y_train == 0)
    expected_signal = int(signal_frac * config["merged_params"]["asimov_batch_size"])
    expected_bg = int((1 / signal_frac) * config["merged_params"]["asimov_batch_size"])
    systematic_uncertainty = config["merged_params"]["systematic_uncertainty"]

    return significance_loss.asimovSignificanceLossInvert(
        expected_signal, expected_bg, systematic_uncertainty
    )


# TODO: add classes for multiclass RNN
# TODO: write summary at the top of all main files

# In results, can then compare:
#       with and without mc dropout
#       with and without asimov loss
#       binary vs multiclass
#       FFN vs RNN vs merged


def main(args):
    mc_dropout = args.mc_dropout
    epochs = args.epochs
    model_type = args.model_type
    data = build_features.load_preprocessed_data(args.all_data)
    class_weights = calculate_class_weights(data["y_train"])
    loss = "binary_crossentropy"

    if args.asimov_loss:
        loss = asimov_loss(data["y_train"])

    model_dict = {
        "FNN": models.FNN_model,
        "RNN": models.RNN_model,
        "merged": models.merged_model,
    }

    if model_type == 'FNN':
        params = config["FNN_params"]
    elif model_type == 'RNN':
        params = config["RNN_params"]
    elif model_type == 'merged':
        params = config['merged_params']

    # makes sure the experiment is reproducible
    reset_random_seeds()

    model = model_dict[model_type](
        use_mc_dropout=mc_dropout,
        loss=loss,
        event_shape=data["event_X_train"].shape[1:],
        object_shape=data["object_X_train"].shape[1:],
        **params,
    )

    # if using wandb
    if args.wandb:
        wandb.init(project="tth-ml", entity="nha")

        wandb.config.epochs = epochs
        wandb.config.mc_dropout = mc_dropout
        wandb.config.all_data = args.all_data
        wandb.config.asimov_loss = args.asimov_loss
        wandb.config.model = type(model)

        for key, value in params.items():
            wandb.config[key] = value

        model.use_wandb()

    reset_random_seeds()

    (
        event_X_train,
        event_X_validation,
        object_X_train,
        object_X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        data["event_X_train"],
        data["object_X_train"],
        data["y_train"],
        stratify=data["y_train"],
        random_state=1,
    )

    if model_type == "RNN":
        X_train = object_X_train
        X_validation = object_X_validation
    elif model_type == "FNN":
        X_train = event_X_train
        X_validation = event_X_validation
    else:
        X_train = [event_X_train, object_X_train]
        X_validation = [event_X_validation, object_X_validation]

    model.train(
        epochs=epochs,
        X_train=X_train,
        X_test=X_validation,
        y_train=y_train,
        y_test=y_validation,
        class_weights=class_weights,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural net")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--wandb", action="store_true", help="Use WandB")
    parser.add_argument(
        "--asimov_loss",
        "-asimov",
        action="store_true",
        help="Use Asimov significance as the loss function",
    )
    parser.add_argument(
        "--all_data",
        "-d",
        action="store_true",
        help="Use all of the available datasets",
    )
    parser.add_argument(
        "--mc_dropout",
        "-mc",
        action="store_true",
        help="Add dropout during testing to calculate model uncertainty",
    )
    parser.add_argument(
        "--model_type",
        "-model",
        "-m",
        choices=("FNN", "RNN", "merged", "multiclass"),
        default="merged",
        help="Which kind of model to train",
    )

    args = parser.parse_args()
    main(args)
