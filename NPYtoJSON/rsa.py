import numpy as np
import os
import json
import gzip

# the function folderlookup is copied from the rsa.py file in Net2Brain
def folderlookup(path):
    """Looks at the available files and returns the chosen one
    Args:
        path (str/path): path to folder
    Returns:
        list: list of files in dir
    """

    files = os.listdir(path)  # Which folders do we have?
    file_sets = []

    for f in files:
        if ".json" not in f and ".DS_Store" not in f:
            if f != ".ipynb_checkpoints":
                file_sets.append(f)

    return file_sets


def loadnpz(path):
    npz = np.load(path, allow_pickle=True)
    key = list(npz.keys())[0]
    return npz[key]


for dataset in ["78images", "92images"]:
    brain_data_path = f"brain_datasets/{dataset}/"

    brain_rdms = folderlookup(brain_data_path)

    for roi in brain_rdms:
        print(roi)
        roi_rdm = loadnpz(brain_data_path + roi)
        roi_flatten_np = roi_rdm.flatten()

        flatten_rdm = list()
        for item in roi_flatten_np:
             flatten_rdm.append(float(item))

        roi_data = {
            "data": list(flatten_rdm),
            "shape": roi_rdm.shape
        }

        json_dumped = json.dumps(roi_data)
        json_bytes = json_dumped.encode("UTF-8")
        gzip_data = gzip.compress(json_bytes)

        with open(f"json_files/brain_datasets/{dataset}/{dataset}_{roi.replace(".npz", "")}.gzip", "wb") as file_output:
                file_output.write(gzip_data)
