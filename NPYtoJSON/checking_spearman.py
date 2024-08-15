from scipy import stats
from scipy.spatial.distance import squareform
import numpy as np
import os
import os.path as op


def sq(x):
    """Converts a square-form distance matrix from a vector-form distance vector

    Args:
        x (numpy array): numpy array that should be vector

    Returns:
        numpy array: numpy array as vector
    """
    return squareform(x, force='tovector', checks=False)


def model_spearman(model_rdm, rdms):
    """Calculate Spearman for model
    Args:
        model_rdm (numpy array): RDM of model
        rdms (list of numpy arrays): RDMs of ROI
    Returns:
        float: Spearman correlation of model and roi
    """

    model_rdm_sq = sq(model_rdm)
    return [stats.spearmanr(sq(rdm), model_rdm_sq)[0] for rdm in rdms]


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

# test data contains distance matrices
test_data_path = "rsa_test_data"
cases = {
    "case1": ["layer3.1.conv2.npz", "layer4.1.bn2.npz"],
    "case2": ["visual.layer4.npz"]
}

brain_data_path = f"brain_datasets/78images/"

brain_rdms = folderlookup(brain_data_path)
for roi in brain_rdms:
    roi_rdm = loadnpz(op.join(brain_data_path, roi))
    
    for case, layers in cases.items():
        for layer in layers:
            # convert npz files
            layer_data = loadnpz(f"{test_data_path}/{case}/{layer}")
            
            if "fmri" in roi:
                spearman_result = model_spearman(layer_data, roi_rdm)
                print("-----------------------------------")
                print(roi)
                print(layer)
                print(spearman_result)
