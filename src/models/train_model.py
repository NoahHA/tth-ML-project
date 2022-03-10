"""
Train a machine learning model for ttH classification

Usage: train_model.py --epochs epochs --all_data --save --model_name model_name.h5

where:
--epochs epochs: number of epochs to run training for (typically 5–50)
--all_data: if added then model will automatically use all backgrounds,
            otherwise you have to specify
--save: if added then model will be saved until --model_name in models folder
--model_name model_name.h5: name of the model, must end with .h5 for keras models
                            and .model for xgboost, defaults to model_test.h5
-h: show help message
"""

import warnings

from tensorflow import get_logger

get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

import argparse
import os
import random

os.environ["PYTHONHASHSEED"] = str(1)  # sets python random seed for reproducibility

import keras.backend as K
import numpy as np
import tensorflow as tf
import wandb
import yaml
from keras.layers import Dropout
from sklearn.utils import class_weight
from src.features import build_features
from src.models import RNN_models, significance_loss
from src.visualization import visualize

config = yaml.safe_load(open("src/config.yaml"))


class MonteCarloDropout(Dropout):
    """Keeps dropout on in testing mode for uncertainty predictions"""

    def call(self, inputs):
        return super().call(inputs, training=True)


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
    expected_signal = int(signal_frac * config["RNN_params"]["batch_size"])
    expected_bg = int((1 / signal_frac) * config["RNN_params"]["batch_size"])
    systemic_uncertainty = config["RNN_params"]["systemic_uncertainty"]

    return significance_loss.asimovSignificanceLossInvert(
        expected_signal, expected_bg, systemic_uncertainty
    )


# TODO: add command line arg for model type with specific options
# TODO: add classes for multiclass RNN
# TODO: write summary at the top of all main files
# TODO: change visualize.py to use mc dropout and plot the mean values
# for e.g. significance with sigmas around it


# TODO: Models I need:
#       merged model asimov loss and mc dropout, 
#       merged model asimov loss w/o mc dropout, 
#       merged model cross-entropy loss and mc dropout, 
#       merged model cross-entropy loss w/o mc dropout,
#       RNN model cross-entropy loss and mc dropout,
#       RNN model cross-entropy loss w/o mc dropout,
#       RNN model asimov loss and mc dropout,
#       RNN model asimov loss w/o mc dropout,
#       FFN model asimov loss and mc dropout,
#       FFN model asimov loss w/o mc dropout
#       FFN model cross-entropy loss and mc dropout,
#       FFN model cross-entropy loss w/o mc dropout,
#       multiclass model cross-entropy loss and mc dropout,
#       multiclass model cross-entropy loss w/o mc dropout,
#       multiclass model asimov loss and mc dropout,
#       multiclass model asimov loss w/o mc dropout,
#       DONE: XGBoost with cross-entropy loss
#       XGBoost with asimov loss


# In results, can then compare:
#       with and without mc dropout
#       with and without asimov loss
#       binary vs multiclass
#       FFN vs RNN vs merged


def main(args):
    epochs = args.epochs
    save_model = args.save
    mc_dropout = args.mc_dropout
    dropout_type = MonteCarloDropout if mc_dropout else Dropout
    model_name = args.model_name
    model_filepath = os.path.join("models", model_name)
    data = build_features.load_preprocessed_data(args.all_data)
    class_weights = calculate_class_weights(data["y_train"])

    if args.asimov_loss:
        loss = asimov_loss(data["y_train"])
    else:
        loss = "binary_crossentropy"

    # makes sure the experiment is reproducible
    reset_random_seeds()

    model = RNN_models.merged_model(
        dropout_type=dropout_type,
        loss=loss,
        event_shape=data["event_X_train"].shape[1:],
        object_shape=data["object_X_train"].shape[1:],
        **config["RNN_params"],
    )

    if save_model:
        model.save(model_filepath)

    # if using wandb
    if not args.nowandb:
        wandb.init(project="tth-ml", entity="nha")

        wandb.config.epochs = epochs
        wandb.config.mc_dropout = mc_dropout
        wandb.config.all_data = args.all_data
        wandb.config.asimov_loss = args.asimov_loss
        wandb.config.model = type(model)

        for key, value in config["RNN_params"].items():
            wandb.config[key] = value

        model.use_wandb()

    reset_random_seeds()

    scores = model.cross_validate(
        epochs=epochs,
        X=[data["event_X_train"], data["object_X_train"]],
        y=data["y_train"],
        class_weights=class_weights,
        cv=3,
    )

    if save_model:
        visualize.make_training_curves(scores)
        visualize.save_plot(model_name, "training_curves")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural net")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--nowandb", action="store_true", help="Don't use WandB")
    parser.add_argument(
        "--asimov_loss",
        action="store_true",
        help="Use Asimov significance as the loss function",
    )
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
        help="Add dropout during testing to calculate model uncertainty",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save model to models folder and save create training curves",
    )

    args = parser.parse_args()
    main(args)
