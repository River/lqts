# LQTSnet Prediction API

Makes predictions for LQTS carrier status and LQTS genotype (type 1 and type 2) based on 12-lead resting ECG data. Containerized flask webapp which exposes an API which takes normalized ECG voltage data as input, and returns the predictions.

## How to Use

### Step 1. Build and run Docker container

Build the Docker container:

```bash
docker build -t lqtsnet .
```

Run the container in the background; the container exposes 8501 port (can map this to whatever port you want on the server). 

```bash
docker run --detach -p 8501:8501 lqtsnet
```

### Step 2. Process ECG data

Extract the ECG lead data from the input file (e.g. XML) and preprocess the lead voltage data. Resample the leads down to 2500 total samples (250 Hz over 10 second recording) and normalize the lead data (mean=0 and std=1). You may use `lqts.extract_ecg_xml.preprocess_leads()` for this. 

LQTSnet expects an input which is an `nparray[float]` with dimensions `(n,2500,8)` corresponding to `n` ECGs (`n`=1 for a single ECG prediction), 2500 samples, and 8 non-augmented leads (I, II, V1, V2, V3, V4, V5, V6). The augmented leads III, AVR, and AVL are not used. Preprocessed data from an example ECG is shown in `test/normalized_ecg.csv`. 

### Step 3. Send predictions

An example script for how to request a prediction from the API is shown in `test/test.py`. With input from the example ECG `test/normalized_ecg.csv`, the output prediction is:

```json
{"prediction":[[1.0,6.238670739950117e-14,7.81060166630404e-13]]}
```