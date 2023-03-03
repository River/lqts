# AI Long QT ECG analysis
Deep Neural Networks in Evaluation of Patients with Congenital Long QT Syndrome from the Surface 12-Lead Electrocardiogram

## Step 0: Install pip packages

Install python packages.

`python -m pip install -r requirements.txt`

## Step 1: Obtain MUSE ECGs

Should be in XML format and the beginning of the files start like this:

```xml
<?xml version="1.0" encoding="ISO-8859-1"?>
<!DOCTYPE RestingECG SYSTEM "restecg.dtd">
```

## Step 2: Convert ECGs into CSV files

Run `python lqtnet/extract_ecg_xml.py`, which converts a folder containing XML ECG files into CSV format, normalizes the voltage data, and resamples all the files (to 2500 samples over the file, 250 Hz over 10 second recording). 

## Step 3: Create metadata file

Create a `metadata` folder and in that create a CSV file with the following columns:

```csv
file,patient_id,ecg_id,id_site,set,lqts_type,dob,age,sex,ethnicity,date,hr,qt,qt_confirmed,qt_prolonged,qc,qc_reason
```

Descriptions for the columns:
- `file`: csv file name (without '.csv' file extension)
- `patient_id`: unique ID for patient (HiRO ID)
- `ecg_id`: unique ID for the ECG file
- `id_site`: HiRO site ID
- `set`: split, `Derivation`, `Internal validation`, or `External validation`
- `lqts_type`: either `Control`, `Type 1`, or `Type 2` based on genetic diagnosis
- `dob`: date of birth, yyyy-mm-dd
- `age`: age (in years)
- `sex`: `Female` or `Male`
- `ethnicity`: used for baseline characteristics and subsequent analysis
- `date`: date of ecg, yyyy-mm-dd
- `hr`: heart rate, for baseline characteristics and subsequent analysis
- `qt_manual`: correct QT interval (in milliseconds)
- `qt_manual_confirmed`: `True` or `False`, was the QT interval manually interpreted?
- `qc`: `True` or `False`, whether ECG passed manual quality control
- `qc_reason` (optional): description of QC issue with ECG

Use `lqtnet.import_metadata.convert_dtypes()` to convert the dtypes for the files for more efficient storage. We also suggest saving the metadata file as `pickle` or `parquet` format after importing it as a pandas `DataFrame`. 

## Step 4: Quality control

Some of the files are missing parts of the leads, excessive noise, wandering leads, are corrupted and don't contain any ECG data, etc. Fill in this data into the above metadata file.

## Step 5: Run model inference

Please see example code below, showing inference for an `External validation` dataset:

```python
import lqtnet

# directory containing normalized CSV files
ECG_SOURCE_DIR = 'ecgs/csv_normalized_2500/'
MODEL_PATH = 'models/XYZ/'

metadata = pd.read_parquet('metadata/example_YYYYmmdd.parquet')
ext_df = metadata.query('set == "External validation" and qc == "Good"')

x_ext = lqtnet.import_ecgs.df_import_csv_to_numpy(ext_df, from_dir=ECG_SOURCE_DIR)
y_ext = lqtnet.import_ecgs.df_to_np_labels(ext_df)

model = lqtnet.train._load_model(MODEL_PATH)

# make predictions - save this output for further analysis
y_extval_pred = model.predict(x_extval)
```
