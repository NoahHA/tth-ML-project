import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def load_data(data_path):
    include_SL = input("Include Semi-Leptonic data? (y/n)\n")
    include_FL = input("Include Fully-Leptonic data? (y/n)\n")
    include_FH = input("Include Fully-Hadronic data? (y/n)\n")

    # loads the dataframes
    higgs_df = pd.read_hdf(os.path.join(data_path, "ttH.hd5"))
    semi_leptonic_df = pd.read_hdf(os.path.join(data_path, "ttsemileptonic.hd5"))
    fully_leptonic_df = pd.read_hdf(os.path.join(data_path, "fully_leptonic.hd5"))
    fully_hadronic_df = pd.read_hdf(os.path.join(data_path, "fully_hadronic.hd5"))

    # labels signal vs background
    higgs_df["signal"] = 1
    semi_leptonic_df["signal"] = 0
    fully_hadronic_df["signal"] = 0
    fully_leptonic_df["signal"] = 0

    # combines the dataframes and randomly shuffles the rows
    full_df = higgs_df
    if include_SL == "y":
        full_df = full_df.append(semi_leptonic_df, ignore_index=True)
    if include_FL == "y":
        full_df = full_df.append(fully_leptonic_df, ignore_index=True)
    if include_FH == "y":
        full_df = full_df.append(fully_hadronic_df, ignore_index=True)

    # removes useless columns
    full_df = shuffle(full_df)
    df = full_df[event_cols + object_cols + ["signal", "xs_weight"]]

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


def preprocess_data():
    ############## LOADING AND PREPROCESSING DATA ##############

    print("LOADING AND PREPROCESSING DATA...")

    data_path = r"data/raw"
    df = load_data(data_path)
    max_jets = df["ncleanedJet"].max()

    # splits data into training and validation
    X, y = df.drop("signal", axis=1), df["signal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1
    )

    # divides training data into object level and event level features
    event_X_train, event_X_test = X_train[event_cols], X_test[event_cols]
    object_X_train, object_X_test = X_train[object_cols], X_test[object_cols]

    # preprocesses the data
    scaler = StandardScaler()
    event_X_train[event_cols] = scaler.fit_transform(event_X_train[event_cols].values)
    event_X_test[event_cols] = scaler.transform(event_X_test[event_cols].values)

    object_X_train = pad_data(object_X_train)
    object_X_test = pad_data(object_X_test)

    object_X_train = expand_lists(object_X_train, max_jets)
    object_X_test = expand_lists(object_X_test, max_jets)

    nz = np.any(object_X_train, -1)
    object_X_train[nz] = scaler.fit_transform(object_X_train[nz])
    nz = np.any(object_X_test, -1)
    object_X_test[nz] = scaler.transform(object_X_test[nz])

    print("FINISHED PREPROCESSING")

    ############## SAVING PROCESSED DATA ##############

    print("SAVING DATA...")

    save_path = r"data/processed"

    np.save(os.path.join(save_path, "object_X_train.npy"), object_X_train)
    np.save(os.path.join(save_path, "object_X_test.npy"), object_X_test)

    event_X_train.to_pickle(os.path.join(save_path, "event_X_train.pkl"))
    event_X_test.to_pickle(os.path.join(save_path, "event_X_test.pkl"))

    y_train.to_pickle(os.path.join(save_path, "y_train.pkl"))
    y_test.to_pickle(os.path.join(save_path, "y_test.pkl"))

    interim_path = r"data/interim"

    X_train.to_pickle(os.path.join(interim_path, "X_train.pkl"))
    X_test.to_pickle(os.path.join(interim_path, "X_test.pkl"))

    print("DATA SAVED")


if __name__ == "__main__":
    preprocess_data()
