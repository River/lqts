import pandas as pd
import numpy as np


def split_test_train_df(df, internal_id_sites, test_frac):
    """
    Split dataframe into internal vs external datasets, and then randomly splits internal dataset
    into testing and training splits, with NONOVERLAPPING unique patients

    Returns df with a new column, "set", which is one of 
    - Derivation
    - Internal validation
    - External validation

    df: input dataframe
    internal_id_sites: list of id_site belonging to internal dataset (rest are assumed to be external)
    test_frac: fraction of patients to split into Internal Validation dataset
    """

    df["set"] = pd.Series(pd.Categorical([], categories=["Derivation",
                                                         "Internal validation",
                                                         "External validation"]))
    
    # Anything where id_site is not in internal_id_site is part of External validation
    df.loc[df.query("id_site not in @internal_id_sites").index, 'set'] = "External validation"

    # List of unique patient_id in Internal dataset
    internal_patient_ids = df.query("id_site in @internal_id_sites").patient_id.unique()
    
    # Randomly choose list of patients for Internal validation
    n_test_patients = round(len(internal_patient_ids)*test_frac)
    val_ids = np.random.choice(internal_patient_ids, n_test_patients, replace=False)

    # Derivation set
    df.loc[df.query("(id_site in @internal_id_sites) and (patient_id not in @val_ids)").index, 'set'] = \
        "Derivation"
    
    # # Internal validation set
    df.loc[df.query("(id_site in @internal_id_sites) and (patient_id in @val_ids)").index, 'set'] = \
        "Internal validation"
    
    return df