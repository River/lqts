import requests
import json
import pandas as pd
import numpy as np

if __name__ == "__main__":
    input_df = pd.read_csv("normalized_ecg.csv", index_col=0)
    input_np = input_df.to_numpy()
    input_np = np.reshape(input_np, (1, 2500, 8))
    json_data = json.dumps(input_np.tolist())

    # send POST request
    url = "http://localhost:8501"
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, data=json_data, headers=headers)

    # Print the response from the server
    print(response.text)