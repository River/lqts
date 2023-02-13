# Modified from https://github.com/hewittwill/ECGXMLReader

import os
import array
import base64
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import signal
import argparse
import untangle


class ECGXMLExtract:
    def __init__(self, path):
        self.path = path
        self.lead_voltages = None
        self.lead_names = None

        try:
            self.ecg = untangle.parse(path)

            self.patient_demographics = self.ecg.RestingECG.PatientDemographics
            self.test_demographics = self.ecg.RestingECG.TestDemographics
            self.resting_ecg_measurements = self.ecg.RestingECG.RestingECGMeasurements

            # Waveform[0] is the "median" beat
            # Waveform[1] is the "rhythm" waveform
            self.waveform = self.ecg.RestingECG.Waveform[1]

            self.lead_voltages = self.makeLeadVoltages()
            self.lead_names = self.lead_voltages.columns

        except Exception as e:
            print("Unable to parse: " + path + " " + str(e))

    def makeLeadVoltages(self):
        num_leads = 0
        leads = pd.DataFrame()

        for lead in self.waveform.LeadData:
            num_leads += 1

            lead_data = lead.WaveFormData.cdata
            lead_b64 = base64.b64decode(lead_data)
            # 'h' indicates each byte represents a signed short int
            # see 'type codes' in the array documentation
            lead_vals = np.array(array.array("h", lead_b64))

            leads[lead.LeadID.cdata] = lead_vals

        return leads

    def getVoltages(self):
        return self.lead_voltages

    def getVoltagesNp(self):
        return self.lead_voltages.to_numpy()


def preprocess_leads(leads, samples):
    """
    Preprocess ECG Data (input: numpy array dim(n,8))
    1. Resample data to 250 Hz (source data is 250-500 Hz)
    2. Normalize values to have mean 0 and std 1
    """

    # scale to mean=0 and std=1
    scalar = StandardScaler(with_mean=True, with_std=True)
    leads_scale = scalar.fit_transform(leads)

    # resample to 2500 samples (10 sec period) using Fourier method
    # see: scipy.signal.resample()
    leads_scale_resamp = signal.resample(leads_scale, samples, axis=0)

    return leads_scale_resamp


def convert_xml_to_csv(source_dir, dest_dir, samples=2500):
    file_names = os.listdir(source_dir)

    for file_name in file_names:
        try:
            ecg = ECGXMLExtract(source_dir + "/" + file_name)
            v = ecg.getVoltagesNp()
            if v is not None:
                v_n = preprocess_leads(v, samples)
                ecg_n = pd.DataFrame(columns=ecg.lead_names, data=v_n)

                # clip off the extension
                save_to_path = dest_dir + "/" + file_name[0:-4] + ".csv"
                ecg_n.to_csv(save_to_path)
                print(f"Saved: {save_to_path}")

        except Exception as e:
            print(f"Error: {e}")


def remove_trailing_slash(path):
    return path[:-1] if path[-1] == "/" else path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", help="path to ecg files (xml)", required=True)
    parser.add_argument("--dest_dir", help="path to save csv files", required=True)
    parser.add_argument(
        "--samples",
        help="number of samples per ecg file (default 2500)",
        default=2500,
        type=int,
    )
    args = parser.parse_args()

    convert_xml_to_csv(
        remove_trailing_slash(args.source_dir),
        remove_trailing_slash(args.dest_dir),
        args.samples,
    )
