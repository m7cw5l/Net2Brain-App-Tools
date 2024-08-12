import json
import gzip
import numpy as np

for hemisphere in ["left", "right"]:
    brain_vertices = np.load(f"brain_models/{hemisphere}/brain_model/brain_coordinates.npy")
    brain_normals = np.load(f"brain_models/{hemisphere}/brain_model/brain_normals.npy")
    brain_faces = np.load(f"brain_models/{hemisphere}/brain_model/brain_faces.npy")

    brain_surface_map = np.load(f"brain_models/{hemisphere}/brain_model/brain_surface.npy")
    brain_surface_map_norm = (brain_surface_map - np.min(brain_surface_map)) / (np.max(brain_surface_map) - np.min(brain_surface_map))

    vertices_array = list()
    normals_array = list()
    faces_array = list()
    surface_array = list()

    for vector in brain_vertices:
        # vertices_array.append([float(vector[0]), float(vector[1]), float(vector[2])])
        vertices_array.append({"x": float(vector[0]),"y": float(vector[1]),"z": float(vector[2])})

    for vector in brain_normals:
        normals_array.append({"x": float(vector[0]),"y": float(vector[1]),"z": float(vector[2])})

    for vector in brain_faces:
        faces_array.append(int(vector[0]))
        faces_array.append(int(vector[1]))
        faces_array.append(int(vector[2]))

    for item in brain_surface_map_norm:
        surface_array.append(float(item))

    json_vertices = json.dumps(vertices_array)
    json_normals = json.dumps(normals_array)
    json_faces = json.dumps(faces_array)
    json_surface = json.dumps(surface_array)

    json_vertices_bytes = json_vertices.encode("UTF-8")
    json_normals_bytes = json_normals.encode("UTF-8")
    json_faces_bytes = json_faces.encode("UTF-8")
    json_surface_bytes = json_surface.encode("UTF-8")

    gzip_vertices = gzip.compress(json_vertices_bytes)
    gzip_normals = gzip.compress(json_normals_bytes)
    gzip_faces = gzip.compress(json_faces_bytes)
    gzip_surface = gzip.compress(json_surface_bytes)

    with open(f"json_files/brain_model/brain_vertices_{hemisphere}.gzip", "wb") as file_output:
        file_output.write(gzip_vertices)

    with open(f"json_files/brain_model/brain_normals_{hemisphere}.gzip", "wb") as file_output:
        file_output.write(gzip_normals)

    with open(f"json_files/brain_model/brain_faces_{hemisphere}.gzip", "wb") as file_output:
        file_output.write(gzip_faces)

    with open(f"json_files/brain_model/brain_surface_{hemisphere}.gzip", "wb") as file_output:
        file_output.write(gzip_surface)

    for roi in ["all", "anatomical", "body", "face", "place", "visual", "word"]:
        color_map = np.load(f"brain_models/{hemisphere}/roi_{roi}_color_map.npy")

        color_array = list()
        for item in color_map:
            color_array.append(float(item))

        json_color = json.dumps(color_array)

        json_color_bytes = json_color.encode("UTF-8")

        gzip_color = gzip.compress(json_color_bytes)

        with open(f"json_files/vertices/roi_{hemisphere}_{roi}_color_map.gzip", "wb") as file_output:
            file_output.write(gzip_color)