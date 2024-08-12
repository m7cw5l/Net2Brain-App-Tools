import numpy as np
from collada import *
from collada import source
from collada import scene

from nilearn import surface

hemisphere = "right"

# load the data
surf_mesh = surface.load_surf_mesh(f"Gifti Data/infl_{hemisphere}.gii.gz")
surf_mesh_coordinates = surf_mesh.coordinates # ndarray
surf_mesh_faces = surf_mesh.faces # ndarray

surf_map = np.load("Numpy Data/lh.all-vertices_fsaverage_space.npy") # ndarray; highlighted regions on brain

surface_data = surface.load_surf_data(f"Gifti Data/sulc_{hemisphere}.gii.gz") # ndarray; texture for brain
print(f"LENGTH COORDINATES: {len(surf_mesh_coordinates)}")
print(f"LENGTH FACES: {len(surf_mesh_faces)}")
print(f"LENGTH TEXTURE: {len(surface_data)}")
print(f"LENGTH SURFACE MAP: {len(surf_map)}")

print(f"MIN: {surface_data.min()}")
print(f"MAX: {surface_data.max()}")

surface_data_norm = (surface_data - np.min(surface_data)) / (np.max(surface_data) - np.min(surface_data))
print(f"MIN: {surface_data_norm.min()}")
print(f"MAX: {surface_data_norm.max()}")
print(surface_data_norm)

# start creating collada object
mesh = Collada()

effect = material.Effect("effect0", [], "phong", diffuse=(0.9,0.9,0.9), specular=(1,1,1))
mat = material.Material("material0", "mymaterial", effect)

mesh.effects.append(effect)
mesh.materials.append(mat)

vert_src = source.FloatSource("surf_mesh_coordinates", surf_mesh_coordinates, ('X', 'Y', 'Z'))
#normal_src = source.FloatSource("surface_data", surface_data_norm, ('X', 'Y', 'Z'))
color_src = source.FloatSource("surf_map", surface_data_norm, ('G'))

geom = geometry.Geometry(mesh, "geometry0", f"brain_{hemisphere}", [vert_src])

input_list = source.InputList()
input_list.addInput(0, 'VERTEX', "#surf_mesh_coordinates")
#input_list.addInput(1, 'COLOR', "#surf_map")

indices = surf_mesh_faces
#indices = np.array(range(0,(len(surf_mesh_coordinates))))

triset = geom.createTriangleSet(indices, input_list, "materialref")
triset.generateNormals()
print(f"COLORS: {triset.sources["COLOR"]}")

np.save(f"{hemisphere}_normals.npy", triset.normal)

geom.primitives.append(triset)
mesh.geometries.append(geom)

if hemisphere == "left":
    rotate_x = scene.RotateTransform(1, 0, 0, -90)
    rotate_y = scene.RotateTransform(0, 1, 0, 150)
    rotate_z = scene.RotateTransform(0, 0, 1, 0)
else:
    rotate_x = scene.RotateTransform(1, 0, 0, -90)
    rotate_y = scene.RotateTransform(0, 1, 0, -20)
    rotate_z = scene.RotateTransform(0, 0, 1, 0)

matnode = scene.MaterialNode("materialref", mat, inputs=[])

geomnode = scene.GeometryNode(geom, [matnode])
node = scene.Node("node0", children=[geomnode], transforms=[rotate_y, rotate_x, rotate_z])

myscene = scene.Scene("myscene", [node])
mesh.scenes.append(myscene)
mesh.scene = myscene

#mesh.write(f'brain_{hemisphere}.dae')