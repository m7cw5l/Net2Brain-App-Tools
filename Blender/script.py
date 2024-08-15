import bpy
import numpy
import math

hemisphere = "right"
roi = "all" # all | body | face | place | word | visual | anatomical
export = True

# define paths
base_import_path = "../GiftiConvert/export"
base_export_path = ""


roi_keys = [
    ("all-vertices", "all"),
    ("floc-bodies", "body"),
    ("floc-faces", "face"),
    ("floc-places", "place"),
    ("floc-words", "word"),
    ("prf-visualrois", "visual"),
    ("streams", "anatomical")
]

context = bpy.context

# create basic brain mesh
vertices = numpy.load(f"{base_import_path}/{hemisphere}/brain_model/brain_coordinates.npy")
faces = numpy.load(f"{base_import_path}/{hemisphere}/brain_model/brain_faces.npy")

vertex_color_data = numpy.load(f"{base_import_path}/{hemisphere}/brain_model/brain_surface.npy")

roi_color_data = numpy.load(f"{base_import_path}/{hemisphere}/roi_{roi}_color_map.npy")

# https://medium.com/@bldevries/using-python-and-custom-data-to-vertex-color-your-blender-model-4fd0d69134a3; 13.10.23 08:08
ob_name = f"brain_{roi}_{hemisphere}"

mesh = bpy.data.meshes.new(name=f"{ob_name}_mesh")

ob = bpy.data.objects.new(ob_name, mesh)

mesh.from_pydata(vertices, [], faces)

bpy.context.scene.collection.objects.link(ob)

# https://blender.stackexchange.com/a/280720; 13.10.2023 08:21
colattr = ob.data.color_attributes.new(
    name=f"brain_{roi}_{hemisphere}_colors",
    type='FLOAT_COLOR',
    domain='POINT',
)

for v_index in range(len(ob.data.vertices)):
    color_value = vertex_color_data[v_index]
    roi_color_value = roi_color_data[v_index]
    
    if roi_color_value == 1:
        r = 223 /255
        g = 0
        b = 211 / 255
    else:
        r = color_value
        g = color_value
        b = color_value
    
    colattr.data[v_index].color = [r, g, b, 1]
    
# scale object
# https://docs.blender.org/api/current/bpy.types.Object.html#bpy.types.Object.scale; 16.10.23 11:48
ob.scale[0] = 0.01
ob.scale[1] = 0.01
ob.scale[2] = 0.01

# create camera
# https://blender.stackexchange.com/a/152577; 16.10.23 11:52
cam = bpy.data.cameras.new(f"camera_brain_{hemisphere}")
cam.lens = 24
cam.lens_unit = "FOV"
cam.angle = math.radians(90)
cam.clip_end = 30.0
cam.sensor_fit = "VERTICAL"
cam.sensor_height = 24.0
cam.sensor_width = 36.0
    
cam_obj = bpy.data.objects.new(f"camera_object_brain_{hemisphere}", cam)

if hemisphere == "right":
    cam_obj.location = (19.0, 0.0, -0.7)
    cam_obj.rotation_euler = (math.radians(90), 0, math.radians(90))
else:
    cam_obj.location = (-19.0, 0.0, -0.7)
    cam_obj.rotation_euler = (math.radians(90), 0, math.radians(-90))
    
context.scene.collection.objects.link(cam_obj)

if export:
    # https://blender.stackexchange.com/a/160069; 16.10.23 09:07
    # add basic Principled BSDF Material
    brain_material = bpy.data.materials.new(f"brain_{roi}_{hemisphere}")
    brain_material.use_nodes = True
    node_tree = brain_material.node_tree
    nodes = node_tree.nodes

    bsdf = nodes.get("Principled BSDF")

    assert(bsdf)

    # connect Color Attribute Color Output (index 0) to Principled BSDF Base Color Input (index 0)
    vcol = nodes.new(type="ShaderNodeVertexColor")
    vcol.layer_name = f"brain_{roi}_{hemisphere}_colors" # the vertex color layer name

    node_tree.links.new(vcol.outputs[0], bsdf.inputs[0])

    # add image texture
    # https://docs.blender.org/api/current/bpy.ops.image.html#bpy.ops.image.new; 16.10.23 09:29
    image = bpy.data.images.new(f"brain_{roi}_{hemisphere}_image", width=2048, height=2048, alpha=True)
    image.generated_color = (1, 1, 1, 1)

    img_tex = nodes.new(type="ShaderNodeTexImage")
    img_tex.name = f"brain_{roi}_{hemisphere}_image_texture"
    img_tex.select = True
    nodes.active = img_tex

    img_tex.image = image

    # create UV Map
    # https://blender.stackexchange.com/a/120807; 16.10.23 09:17
    context.view_layer.objects.active = ob
    ob.select_set(True)

    brain_uv =  ob.data.uv_layers.get("BrainUV")
    if not brain_uv:
        brain_uv = ob.data.uv_layers.new(name="BrainUV")
    brain_uv.active = True
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT') # for all faces
    bpy.ops.uv.smart_project(angle_limit=math.radians(50), island_margin = 0.0)

    # assign material to object
    ob.active_material = brain_material

    # bake Color Attribute to UV Map
    filepath_bake = f"{base_export_path}/textures/brain_{roi}_{hemisphere}_texture.png"

    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.bake(type="DIFFUSE", width=2048, height=2048, margin=5, margin_type="ADJACENT_FACES", uv_layer="BrainUV", save_mode="EXTERNAL", use_clear=True)

    # export texture image
    image.save_render(filepath=filepath_bake)

    # change link from VertexColor to ImageTexture
    node_tree.links.new(img_tex.outputs[0], bsdf.inputs[0])

    # export USDZ File
    filepath_usdz = f"{base_export_path}/models/brain_{roi}_{hemisphere}.usdz"
    bpy.ops.wm.usd_export(filepath=filepath_usdz)

    # delete all generated items
    # https://blender.stackexchange.com/a/75439; 16.10.23 11:23
    bpy.data.objects.remove(ob, do_unlink=True)

    bpy.data.cameras.remove(cam)
    bpy.data.images.remove(image)
    bpy.data.materials.remove(brain_material)
    bpy.data.meshes.remove(mesh)