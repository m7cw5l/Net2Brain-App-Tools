import numpy as np
import csv
import json
import gzip

def loadnpz(path):
    npz = np.load(path, allow_pickle=True)
    key = list(npz.keys())[0]
    return npz[key]

# test data contains distance matrices
test_data_path = "rsa_test_data"
cases = {
    "case1": ["layer3.1.conv2.npz", "layer4.1.bn2.npz"],
    "case2": ["visual.layer4.npz"]
}

for case, file_list in cases.items():
    # convert results csv
    with open(f"{test_data_path}/{case}/results.csv", newline='') as csvfile:
        results_reader = csv.DictReader(csvfile)
        """
        Rows: ROI | Layer | Model | R2 | %R2 | Significance | SEM | LNC | UNC
        """
        result_data = list()
        for row in results_reader:
            data = {
                "roi": row["ROI"][4:],
                "layer": row["Layer"][4:-4],
                "model": row["Model"],
                "r2": float(row["R2"]),
                "significance": float(row["Significance"]),
                "sem": float(row["SEM"])
            }
            #print(data)
            result_data.append(data)

        json_dumped = json.dumps(result_data)
        json_bytes = json_dumped.encode("UTF-8")
        gzip_data = gzip.compress(json_bytes)

        with open(f"json_files/rsa_test_data/{case}/results_{case}.gzip", "wb") as file_output:
            file_output.write(gzip_data)

    for file in file_list:
        # convert npz files
        layer_distance = loadnpz(f"{test_data_path}/{case}/{file}")
        #print(layer_distance.shape)
        model_flatten_np = layer_distance.flatten()

        flatten_layer = list()
        for item in model_flatten_np:
             flatten_layer.append(float(item))

        layer_data = {
            "data": list(flatten_layer),
            "shape": layer_distance.shape
        }

        json_dumped = json.dumps(layer_data)
        json_bytes = json_dumped.encode("UTF-8")
        gzip_data = gzip.compress(json_bytes)

        with open(f"json_files/rsa_test_data/{case}/{file.replace(".npz", "")}.gzip", "wb") as file_output:
            file_output.write(gzip_data)
