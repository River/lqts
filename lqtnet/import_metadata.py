import pandas as pd


def convert_dtypes(df):
    """
    Convert dtypes of pandas df for more efficient storage
    """
    df.patient_id = df.patient_id.astype("Int64")
    df.ecg_id = df.ecg_id.astype("Int64")
    df.id_site = df.id_site.astype("category")
    df.lqts_type = df.lqts_type.astype("category")
    df.dob = pd.to_datetime(df.dob, format="%Y-%m-%d")
    df.age = df.age.astype("Int64")
    df.sex = df.sex.astype("category")
    df.ethnicity = df.ethnicity.astype("category")
    df.date = pd.to_datetime(df.date, format="%Y-%m-%d")
    df.hr = df.hr.astype("Int64")
    df.qtc_manual = df.qt.astype("Int64")
    df.qtc_manual_confirmed = df.qt_confirmed.astype("category")
    df.qc = df.qc.astype("category")
    df.qc_reason = df.qc_reason.astype("category")
    return df
