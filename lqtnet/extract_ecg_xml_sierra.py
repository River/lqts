import os
import pandas as pd
import lqtnet.sierraecg
import argparse


class ECGXMLExtract:
    def __init__(self, path):
        self.path = path
        self.lead_voltages = None

        try:
            ecg = lqtnet.sierraecg.read_file(path)
            xml_leads = ecg.leads

            leads_df = pd.DataFrame()

            for lead in xml_leads:
                leads_df[lead.label] = lead.samples

            leads_df.drop(labels=["III", "aVR", "aVL", "aVF"], axis=1, inplace=True)
            self.lead_voltages = leads_df

        except Exception as e:
            print("Unable to parse: " + path + " " + str(e))

    def getVoltages(self, clip_end=False):
        # clip_end: clip last 200 signals from the ecg (in Stollery files this is calibration)
        if clip_end:
            return self.lead_voltages.iloc[:-200, :]
        else:
            return self.lead_voltages


def convert_xml_to_csv(source_dir, dest_dir):
    file_names = os.listdir(source_dir)

    for file_name in file_names:
        # skip .DS_Store in directories
        if file_name == ".DS_Store":
            continue

        # read the xml file
        ecg = ECGXMLExtract(source_dir + "/" + file_name)
        v = ecg.getVoltages(clip_end=True)

        # if reading the file was successful
        if v is not None:
            # clip off the .xml extension and save as .csv file
            v.to_csv(dest_dir + "/" + file_name[0:-4] + ".csv")
            print(f"Converted: {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", help="path to ecg files (xml)")
    parser.add_argument("--dest_dir", help="path to save csv files", required=True)
    args = parser.parse_args()

    convert_xml_to_csv(args.source_dir, args.dest_dir)
