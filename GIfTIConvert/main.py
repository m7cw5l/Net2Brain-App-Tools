import numpy as np
from nilearn import surface

roi_keys = [
    ("all-vertices", "all"),
    ("floc-bodies", "body"),
    ("floc-faces", "face"),
    ("floc-places", "place"),
    ("floc-words", "word"),
    ("prf-visualrois", "visual"),
    ("streams", "anatomical")
]

hemispheres = ["left", "right"]

for hemisphere in hemispheres:
    # load the data
    surf_mesh = surface.load_surf_mesh(f"import/{hemisphere}/brain_model/infl_{hemisphere}.gii.gz")
    surf_mesh_coordinates = surf_mesh.coordinates # ndarray
    surf_mesh_faces = surf_mesh.faces # ndarray

    surface_data = surface.load_surf_data(f"import/{hemisphere}/brain_model/sulc_{hemisphere}.gii.gz") # ndarray; texture for brain
    surface_data_norm = (surface_data - np.min(surface_data)) / (np.max(surface_data) - np.min(surface_data))

    export_base_path = f"export/{hemisphere}"
    export_brain_bath = f"{export_base_path}/brain_model"

    # save brain mesh
    np.save(f"{export_brain_bath}/brain_coordinates.npy", surf_mesh_coordinates)
    np.save(f"{export_brain_bath}/brain_faces.npy", surf_mesh_faces)

    # save brain surface data
    np.save(f"{export_brain_bath}/brain_surface.npy", surface_data_norm)

    for roi_import, roi_export in roi_keys:
        #load roi data
        surf_map = np.load(f"import/{hemisphere}/{hemisphere[0]}h.{roi_import}_fsaverage_space.npy") # ndarray; highlighted regions on brain

        # save roi color data
        np.save(f"{export_base_path}/roi_{roi_export}_color_map.npy", surf_map)