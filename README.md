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

Run `lqtnet.extract_ecg_xml`, which converts a folder containing XML ECG files into CSV format, normalizes the voltage data, and resamples all the files (to 2500 samples over the file, 250 Hz over 10 second recording). 

## Step 3: Create metadata file

Create a `metadata` folder and in that create a CSV file with the following columns:

```csv
file,patient_id,ecg_id,id_site,set,batch,lqts_type,dob,age,sex,ethnicity,date,hr,qt,qt_confirmed,qt_prolonged,Qc,Qc_reason
```

Descriptions for the columns:
- `file`: file name (minus file extension)
- `patient_id`: unique ID for patient (HiRO ID)
- `ecg_id`: unique ID for the ECG file
- `id_site`: HiRO site ID
- `set`: `Derivation`, `Internal validation`, or `External validation`
- `batch`: name for batch of ECGs (only used for debugging)
- `lqts_type`: either `Control`, `Type 1`, or `Type 2` based on genetic diagnosis
- `dob`: date of birth, yyyy-mm-dd
- `age`: age (in years)
- `sex`: `Female` or `Male`
- `ethnicity`: used for baseline characteristics and subsequent analysis
- `date`: date of ecg, yyyy-mm-dd
- `hr`: heart rate, for baseline characteristics and subsequent analysis
- `qt`: correct QT interval (in milliseconds)
- `qt_confirmed`: `True` or `False`, was the QT interval manually interpreted?
- `qt_prolonged`: `True` or `False`, QTc >460 ms (men) or >470 ms (women), for baseline characteristics and subsequent analysis
- `Qc`: `True` or `False`, whether ECG passed manual quality control
- `Qc_reason` (optional): description of QC issue with ECG

An example row:
```csv
005430_00032957,5430.0,32957.0,2.0,Derivation,2021_nov,Control,1947-07-15,75.0,Female,White,2019-03-20,68.0,410.0,True,False,Good,
```

Use `lqtnet.import_metadata.convert_dtypes()` to convert the dtypes for the files for more efficient storage. Save the final metadata file as `pickle` or `parquet`. 

## Step 4: Quality control

**Very important!** Some of the files are missing parts of the leads, excessive noise, wandering leads, are corrupted and don't contain any ECG data, etc. Fill in this data into the above metadata file.

## Step 5: Import data and run inference

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
