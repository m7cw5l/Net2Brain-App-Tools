import torch.nn as nn
from PIL import Image
from collections import namedtuple
from urllib.request import urlopen
from functools import lru_cache
import coremltools as ct
import torch
import torchvision
import torchvision.models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

# Ensure torchvision version is correct
assert torchvision.__version__ == '0.16.1', "Please update torchvision to 0.16.1"


my_models = {
    'alexnet':  [
            "features.0",
            "features.3",
            "features.6",
            "features.8",
            "features.10"
        ],
    'resnet18': [
            "layer1",
            "layer2",
            "layer3",
            "layer4"
        ],
    'resnet34': [
            "layer1",
            "layer2",
            "layer3",
            "layer4"
        ],
    'resnet50': [
            "layer1",
            "layer2",
            "layer3",
            "layer4"
        ],
    'vgg11': [
            "features.0",
            "features.3",
            "features.8",
            "features.13",
            "features.18"
        ],
    'vgg13': [
            "features.2",
            "features.7",
            "features.12",
            "features.17",
            "features.22"
        ],
}


class Wrapper(nn.Module):
    def __init__(self, module, nodes):
        super().__init__()
        self.module = module
        self.clean_nodes = {node: node.replace(".", "_") for node in nodes}
        self.output = namedtuple("output", self.clean_nodes.values())

    def forward(self, x):
        out = self.module(x)
        # replace names with clean names
        out = {self.clean_nodes[k]: v for k, v in out.items()}
        return self.output(**out)


@lru_cache(maxsize=1)
def get_dummy_image():
    # Load an example image
    image_link = "https://upload.wikimedia.org/wikipedia/commons/8/8e/Hauskatze_langhaar.jpg"
    return Image.open(urlopen(image_link)).convert("RGB")


def create_model(model_name):
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    transform = weights.transforms()
    full_model = torchvision.models.get_model(model_name, weights=weights)

    layers = my_models[model_name]
    print(transform.crop_size)
    print(f"{transform.mean=}, {transform.std=}")

    # Create feature extractor
    model = create_feature_extractor(full_model, return_nodes=layers)
    model = Wrapper(model, layers)
    model.eval()

    # Test feature shapes
    dummy_input = torch.randn(1, 3, transform.crop_size[0],
                              transform.crop_size[0])  # (Batch, Channel, Height, Width)

    print("Test model with dummy input")
    out = model(dummy_input)
    for name, value in zip(out._fields, out):
        print(f"Layer: '{name}' | Shape: {tuple(value.shape)}")

    # Compile model
    ts_model = torch.jit.trace(model, dummy_input)

    # Convert to mlpackage
    mlpackage_obj = ct.convert(
        ts_model,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name='image', shape=dummy_input.shape, channel_first=True, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name=name) for name in out._fields]
    )

    ######### Predict #########
    #image_t = transform(get_dummy_image()).unsqueeze(0)  # Shape (1, 3, h, w)
    # https://pillow.readthedocs.io/en/stable/reference/Image.html; 30.11.2023 15:11
    image_t = get_dummy_image().resize((224, 224))


    pred = mlpackage_obj.predict({'image': image_t})
    for i, (k, v) in enumerate(pred.items()):
        assert k in out._fields
        assert v.shape == getattr(out, k).shape

    return mlpackage_obj


if __name__ == '__main__':
    for model_name in my_models.keys():
        print(f"Create model for {model_name}")
        mlpackage_obj = create_model(model_name)
        mlpackage_obj.save(f"coreml/{model_name}.mlpackage")
        break
