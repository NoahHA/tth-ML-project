"""Script to preprocess the raw datasets. Returns a dicitonary containing:
    "event_X_train": DataFrame of event level variables for training
    "event_X_test": DataFrame of event level variables for testing
    "object_X_train": Numpy array of object level variables for training
    "object_X_test": Numpy array of object level variables for testing
    "y_train": Series of y labels for training
    "y_testing": Series of y labels for testing
    
    The dictionary is saved in data/processed as a .pkl file. Data is not scaled
    as this must be done within the cross-validation loop to prevent data leakage.

    Usage: python build_features.py --all_data
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

pd.options.mode.chained_assignment = None
config = yaml.safe_load(open("src/config.yaml"))

event_cols = config["data"]["event_cols"]
object_cols = config["data"]["object_cols"]


def load_dataset(dataset):
    """ "loads a given hdf file"""
    data_path = config["paths"]["raw_path"]
    dataset_path = os.path.join(data_path, dataset)
    df = pd.read_hdf(dataset_path)

    return df


def load_data(load_all: bool = True):
    backgrounds = ["ttsemileptonic.hd5", "fully_leptonic.hd5", "fully_hadronic.hd5"]
    signal = "ttH.hd5"

    if not load_all:
        for background in backgrounds[:]:
            include = input(f"Include {background} dataset? (y/n)\n")
            if include != "y":
                backgrounds.remove(background)

    full_df = load_dataset(signal)
    full_df["signal"] = 1

    for background in backgrounds:
        bg_df = load_dataset(background)
        bg_df["signal"] = 0
        full_df = full_df.append(bg_df, ignore_index=True)

    shuffled_df = shuffle(full_df)
    useful_cols = config["data"]["useful_cols"] + event_cols + object_cols
    final_df = shuffled_df[useful_cols]

    return final_df


def unskew_data(df):
    untransformed_cols = config["data"]["untransformed_cols"]
    transformed_cols = list(set(event_cols) - set(untransformed_cols))

    # takes the log of each column to remove skewness
    for col_name in event_cols:
        if col_name in transformed_cols:
            df[col_name] = np.log(df[col_name])

    return df


def pad_data(df):
    # pads input sequences with zeroes so they're all the same length
    for col in object_cols:
        df[col] = sequence.pad_sequences(
            df[col].values, padding="post", dtype="float32"
        ).tolist()

    return df


def expand_lists(df, max_jets):
    temp_df = np.ndarray(shape=(df.shape[0], max_jets, len(df.columns)))

    for i, col_name in enumerate(object_cols):
        col = df[col_name]
        for j, event in enumerate(col):
            for k, item in enumerate(event):
                temp_df[j][k][i] = item

    return temp_df


def split_data(df: pd.DataFrame):
    """splits data into training and validation"""
    X, y = df.drop("signal", axis=1), df["signal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1
    )

    # divides training data into object level and event level features
    event_X_train, event_X_test = X_train[event_cols], X_test[event_cols]
    object_X_train, object_X_test = X_train[object_cols], X_test[object_cols]

    all_data = {
        "X_train": X_train,
        "X_test": X_test,
        "event_X_train": event_X_train,
        "event_X_test": event_X_test,
        "object_X_train": object_X_train,
        "object_X_test": object_X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    return all_data


def subtract_leading_jet_phi(df):
    """Subtracts the leading  jet phi from all other jets"""
    phi_index = object_cols.index("cleanedJet_phi")
    leading_jet_phis = df[:, 0, phi_index]
    df[:, :, phi_index] = np.abs(df[:, :, phi_index] - leading_jet_phis[:, None])

    return df


def preprocess_data(all_data):
    max_jets = all_data["X_train"]["ncleanedJet"].max()

    # unskews data
    all_data["event_X_train"] = unskew_data(all_data["event_X_train"])
    all_data["event_X_test"] = unskew_data(all_data["event_X_test"])

    # pads object level data
    all_data["object_X_train"] = pad_data(all_data["object_X_train"])
    all_data["object_X_test"] = pad_data(all_data["object_X_test"])

    # expands object level data
    all_data["object_X_train"] = expand_lists(all_data["object_X_train"], max_jets)
    all_data["object_X_test"] = expand_lists(all_data["object_X_test"], max_jets)

    # subtracts leading phi from object data
    all_data["object_X_train"] = subtract_leading_jet_phi(all_data["object_X_train"])
    all_data["object_X_test"] = subtract_leading_jet_phi(all_data["object_X_test"])

    return all_data


def save_data(data):
    save_path = config["paths"]["processed_path"]

    with open(os.path.join(save_path, "processed_data.pickle"), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_preprocessed_data(use_all_data=True):
    """Loads and returns data to be fed to the model
    Args:
        use_all_data (Bool): Whether or not to use all the background data,
            if False then data needs to be reprocessed using process_data()

    Returns:
        dict: data dictionary containing:
            X_train, X_test,
            event_X_train, event_X_test,
            object_X_train, object_X_test,
            y_train, y_test,
    """
    load_path = config["paths"]["processed_path"]

    if not use_all_data:
        df = load_data(load_all=False)
        all_data = split_data(df)
        preprocessed_data = preprocess_data(all_data)
        save_data(preprocessed_data)

    with open(os.path.join(load_path, "processed_data.pickle"), "rb") as handle:
        combined_data = pickle.load(handle)

    return combined_data


def scale_event_data(event_X_train, event_X_test):
    """scales event level data"""
    scaler = StandardScaler()
    event_X_train.loc[:, event_X_train.columns] = scaler.fit_transform(
        event_X_train.loc[:, event_X_train.columns].values
    )
    event_X_test.loc[:, event_X_train.columns] = scaler.transform(
        event_X_test.loc[:, event_X_train.columns].values
    )
    return (event_X_train, event_X_test)


def scale_object_data(object_X_train, object_X_test):
    # scales object data
    scaler = StandardScaler()
    nz = np.any(object_X_train, -1)
    object_X_train[nz] = scaler.fit_transform(object_X_train[nz])
    nz = np.any(object_X_test, -1)
    object_X_test[nz] = scaler.transform(object_X_test[nz])

    return (object_X_train, object_X_test)


def main(args):
    df = load_data(args.all_data)
    all_data = split_data(df)
    preprocessed_data = preprocess_data(all_data)
    save_data(preprocessed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and Save the Model Dataset"
    )
    parser.add_argument(
        "--all_data",
        action="store_true",
        help="Use all of the available datasets",
    )

    args = parser.parse_args()
    main(args)
