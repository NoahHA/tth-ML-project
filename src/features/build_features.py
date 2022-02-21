import os
import pickle

# ignores all warnings
import warnings

import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")


event_cols = [
    "BiasedDPhi",
    "DiJet_mass",
    "HT",
    "InputMet_InputJet_mindPhi",
    "InputMet_pt",
    "MHT_pt",
    "MinChi",
    "MinOmegaHat",
    "MinOmegaTilde",
    "ncleanedBJet",
    "ncleanedJet",
]

object_cols = [
    "cleanedJet_pt",
    "cleanedJet_area",
    "cleanedJet_btagDeepB",
    "cleanedJet_chHEF",
    "cleanedJet_eta",
    "cleanedJet_mass",
    "cleanedJet_neHEF",
    "cleanedJet_phi",
]


def add_background(data_path, background, full_df):
    bg_path = os.path.join(data_path, f"{background}.hd5")
    bg_df = pd.read_hdf(bg_path)
    bg_df["signal"] = 0

    return full_df.append(bg_df, ignore_index=True)


def load_data():
    include_SL = input("Include Semi-Leptonic data? (y/n)\n")
    include_FL = input("Include Fully-Leptonic data? (y/n)\n")
    include_FH = input("Include Fully-Hadronic data? (y/n)\n")

    data_path = r"data/raw"
    higgs_df = pd.read_hdf(os.path.join(data_path, "ttH.hd5"))
    higgs_df["signal"] = 1

    full_df = higgs_df
    if include_SL == "y":
        full_df = add_background(data_path, "ttsemileptonic", full_df)
    if include_FL == "y":
        full_df = add_background(data_path, "fully_leptonic", full_df)
    if include_FH == "y":
        full_df = add_background(data_path, "fully_hadronic", full_df)

    # removes useless columns
    full_df = shuffle(full_df)
    useful_cols = ["signal", "xs_weight"]
    df = full_df[event_cols + object_cols + useful_cols]

    return df


def unskew_data(df):
    untransformed_cols = ["ncleanedBJet", "ncleanedJet", "BiasedDPhi", "signal"]
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


def split_data(df):
    # splits data into training and validation
    X, y = df.drop("signal", axis=1), df["signal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1
    )
    interim_path = r"data/interim"
    X_train.to_pickle(os.path.join(interim_path, "X_train.pkl"))
    X_test.to_pickle(os.path.join(interim_path, "X_test.pkl"))

    # divides training data into object level and event level features
    event_X_train, event_X_test = X_train[event_cols], X_test[event_cols]
    object_X_train, object_X_test = X_train[object_cols], X_test[object_cols]

    event_data = {"event_X_train": event_X_train, "event_X_test": event_X_test}
    object_data = {"object_X_train": object_X_train, "object_X_test": object_X_test}
    y_data = {"y_train": y_train, "y_test": y_test}

    return (event_data, object_data, y_data)


def preprocess_data():
    print("LOADING AND PREPROCESSING DATA...")

    df = load_data()
    max_jets = df["ncleanedJet"].max()
    event_data, object_data, y_data = split_data(df)

    scaler = StandardScaler()
    event_data["event_X_train"][event_cols] = scaler.fit_transform(
        event_data["event_X_train"][event_cols].values
    )
    event_data["event_X_test"][event_cols] = scaler.transform(
        event_data["event_X_test"][event_cols].values
    )

    for data in object_data.keys():
        object_data[data] = pad_data(object_data[data])
        object_data[data] = expand_lists(object_data[data], max_jets)

    nz = np.any(object_data["object_X_train"], -1)
    object_data["object_X_train"][nz] = scaler.fit_transform(
        object_data["object_X_train"][nz]
    )
    nz = np.any(object_data["object_X_test"], -1)
    object_data["object_X_test"][nz] = scaler.transform(
        object_data["object_X_test"][nz]
    )

    save_data(event_data, object_data, y_data)


def save_data(event_data, object_data, y_data):
    print("SAVING DATA...")
    save_path = r"data/processed"

    combined_data = {
        "event_X_train": event_data["event_X_train"],
        "event_X_test": event_data["event_X_test"],
        "object_X_train": object_data["object_X_train"],
        "object_X_test": object_data["object_X_test"],
        "y_train": y_data["y_train"],
        "y_test": y_data["y_test"],
    }

    with open(os.path.join(save_path, "processed_data.pickle"), "wb") as handle:
        pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("DATA SAVED")


def load_preprocessed_data(use_all_data=True):
    """Loads and returns data to be fed to the model
    Args:
        use_all_data (Bool): Whether or not to use all the background data,
            if False then data needs to be reprocessed using process_data()

    Returns:
        dict: data dictionary containing:
            event_X_train,
            event_X_test,
            object_X_train,
            object_X_test,
            y_train,
            y_test
    """
    load_path = r"data/processed"

    if not use_all_data:
        preprocess_data()

    with open(os.path.join(load_path, "processed_data.pickle"), "rb") as handle:
        combined_data = pickle.load(handle)

    return combined_data


if __name__ == "__main__":
    preprocess_data()
