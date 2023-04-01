import pandas as pd
import numpy as np


def df_import_csv_to_numpy(df, from_dir):
    """
    Reads specified ecg csv files from directory

    Input:
    - df: DataFrame containing file column (excluding extension .csv)
    - from_dir: path to dir containing files (include trailing '/')

    Output:
    numpy array of shape, (# files, # leads, # voltage samples)
    """
    file_names = df.file.to_list()
    ecgs = []
    error = []
    for f in file_names:
        ecg = pd.read_csv(f"{from_dir}{f}", index_col=0).to_numpy()
        ## If shape == (2500, 8) then append otehrwise don't append
        if ecg.shape == (2500, 8):
            ecgs.append(ecg)
        else:
            error.append(f)


    ecgs = np.stack(ecgs)
    return ecgs, error


def df_to_np_labels(df):
    """
    Converts labels from df into one-hot encoding ready for training
    ** Ignores anything other than control/type 1/type 2 (e.g. ignores type 3!)

    Input:
    - df: DataFrame containing lqts_type column

    Output:
    numpy array of shape, (# files, 3)
    """
    labels = df.lqts_type.to_list()

    labels_oh = []
    for l in labels:
        if (l == "Control") or (l == "Control (unconfirmed)"):
            labels_oh.append([1.0, 0.0, 0.0])
        elif l == "Type 1":
            labels_oh.append([0.0, 1.0, 0.0])
        elif l == "Type 2":
            labels_oh.append([0.0, 0.0, 1.0])
    labels_oh = np.stack(labels_oh)
    return labels_oh
