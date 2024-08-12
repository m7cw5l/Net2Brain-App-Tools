import json
import gzip
import numpy as np

lh_fmri = np.load("brain_models/fmri/lh_training_fmri.npy")
rh_fmri = np.load("brain_models/fmri/rh_training_fmri.npy")

#image_index = 0

#hemisphere = "left"
#roi = "visual"

def save_surface(surface_np):
    surface_array = list()

    for item in surface_np:
        surface_array.append(float(item))

    json_surface = json.dumps(surface_array)
    json_surface_bytes = json_surface.encode("UTF-8")
    gzip_surface = gzip.compress(json_surface_bytes)

    with open(f"json_files/fmri/roi_{hemisphere}_{roi}_color_map_image{(image_index + 1):05}.gzip", "wb") as file_output:
            file_output.write(gzip_surface)


for hemisphere in ["left", "right"]:
    for roi in ["all", "anatomical", "body", "face", "place", "visual", "word"]:
        for image_index in range(0, 20):
            if roi == "all":
                fsaverage_all_vertices = np.load(f"brain_models/{hemisphere}/roi_all_color_map.npy")

                fsaverage_response = np.zeros(len(fsaverage_all_vertices))
                if hemisphere == 'left':
                    fsaverage_response[np.where(fsaverage_all_vertices)[0]] = lh_fmri[image_index]
                elif hemisphere == 'right':
                    fsaverage_response[np.where(fsaverage_all_vertices)[0]] = rh_fmri[image_index]

                save_surface(fsaverage_response)
            else:
                roi_classes = {
                    "visual": "prf-visualrois",
                    "body": "floc-bodies",
                    "face": "floc-faces",
                    "place": "floc-places",
                    "word": "floc-words",
                    "anatomical": "streams"
                }

                roi_classes = {
                    "visual": "V1v",
                    "body": "EBA",
                    "face": "OFA",
                    "place": "OPA",
                    "word": "OWFA",
                    "anatomical": "early"
                }

                roi_class = roi_classes[roi]

                # Load the ROI brain surface maps
                challenge_roi_class_dir = f"brain_models/{hemisphere}/roi_{roi}_color_map_challenge.npy"
                fsaverage_roi_class_dir = f"brain_models/{hemisphere}/roi_{roi}_color_map.npy"
                roi_map_dir = f"brain_models/fmri/mapping_{roi}.npy"
                challenge_roi_class = np.load(challenge_roi_class_dir)
                fsaverage_roi_class = np.load(fsaverage_roi_class_dir)
                roi_map = np.load(roi_map_dir, allow_pickle=True).item()

                # Select the vertices corresponding to the ROI of interest
                roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi_class)]
                challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
                fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)

                # Map the fMRI data onto the brain surface map
                fsaverage_response = np.zeros(len(fsaverage_roi))
                if hemisphere == 'left':
                    fsaverage_response[np.where(fsaverage_roi)[0]] = \
                        lh_fmri[image_index,np.where(challenge_roi)[0]]
                elif hemisphere == 'right':
                    fsaverage_response[np.where(fsaverage_roi)[0]] = \
                        rh_fmri[image_index,np.where(challenge_roi)[0]]
                    
                save_surface(fsaverage_response)