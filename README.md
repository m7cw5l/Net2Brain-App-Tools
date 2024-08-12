# Additional Tools for Net2Brain App

This repository contains various tools that where used to convert data during the development of the Net2Brain iOS App.  
Some of them are needed to expand the app with new data.

## Data Sources
The GIfTI-files for brain 3D model are installed by `nilearn` on your computer. On macOS the files are located at `~/nilearn_data/fsaverage/`.

The Numpy-files were included in the Algonauts 2023 Challenge Data. For the app only the data from Subject 01 was used.
It included the general surface maps for the different ROIs in the `/roi_masks`-folder and the fMRI data for the different images in the `/training_split/training_fmri/`-folder

## Tools Overview
The following chapter gives an overview over the different tools developed and used during the creation of the app.
### GIfTIConvert
This tool loads GIfTI-files (`.gii.gz`) and converts them to Numpy-files (`.npy`).  
The files converted contain the brain model for the left and right hemisphere and the surface maps for the different ROIs and form the basis for some of the other tools.  
The GIfTI-files are installed by `nilearn` on your computer. On macOS the files are located at `~/nilearn_data/fsaverage/`.  
The Numpy-files were included in the Algonauts 2023 Challenge Data. For the app only the data from Subject 01 was used.

### Collada
This tool contains a script (`main.py`) that originally converted the 3D brain data from `.gii.gz`-files into a 3D model in the Collada-Format (`.dae`). This functionality is not longer used for the final app, because 3D models are generated on-demand on-device.  
Currently the script is used to export the 3D model normals as a `.npy`-file to use them in the app.

### Blender
The Blender Folder contains the Python-Script (`script.py`) and a Blender-File (`Brain Converter.blend`) with the script embedded. It was used for creating 3D brain model files in the `.usdz`-format, but is not used for the final version of the app, because 3D models are generated on-demand on-device.  
The script loads the input files from the `/GIfTIConvert/export/` folder

### NPYtoJSON
This tool contains different scripts with different purposes. All scripts convert `.npy`- or `.npz`-files to gzip-compressed json-files (`.gzip`).
- `main.py`: converts 3D brain vertices, normals, faces and surface data and saves it to `/json_files/brain_model/`.  
Additionally the surface maps for the different ROIs are converted and saved to `/json_files/vertices/`.
- `fmri.py`: loads fMRI data for every ROI and exports a subset of it (first 20 images) as single files to `/json_files/fmri/`.
- `rsa.py`: loads brain RDM data for RSA from `.npz`-files and converts it to gzip-compressed json-files in `/json_files/brain_datasets/`.
- `rsa_test.py`: loads the test data from `/rsa_test_data/` and saves converted data to `/json_files/rsa_test_data/`.
- `checking_spearman.py`: used to load test data from `/rsa_test_data/` and using it to perform Spearman Correlation on it to compare the results to the app and check if the Spearman-Implementation in the app works.

### TorchToCoreML
Script that converts PyTorch-models to CoreML Packages (`.mlpackage`). The resulting package files in `/coreml/` can be added to the app and used for prediction.  
This script was kindly provided by M. Sc. Timothy Schaumlöffel.