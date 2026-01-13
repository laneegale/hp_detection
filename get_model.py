import timm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transformers import AutoImageProcessor, AutoModel
from trident.patch_encoder_models import encoder_factory

AVAILABLE_MODEL = [
    # "virchow2",
    # "ctranspath",
    # "hoptimus0",
    # "hoptimus1",
    # "uni_v2",
    # "uni_v1",
    # "musk",
    # "conch_v15"
    "resnet50"
]
        
from torchvision.models import resnet50, ResNet50_Weights

# Load the weights and the model

def get_model_and_transform(model_name: str):
    """
    Returns (model, transform) for the given encoder name.

    Supported model_name values:
      - "virchow2"
      - "ctranspath"
      - "hoptimus0"
      - "hoptimus1"
      - "uni_v2"
      - "uni_v1"
      - "musk"
      - "conch_v15"
      - "mstar"       # timm.create_model('hf-hub:Wangyh/mSTAR', ...)
      - "hibou-l"     # transformers histai/hibou-L

    Raises:
      ValueError if model_name is not recognized.
    """
    name = model_name.lower()

    # your existing encoder_factory-based models:
    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)

        transform = transforms.Compose([
            transforms.Resize(256),              # shorter side = 256
            transforms.CenterCrop(224),          # crop to 224×224
            transforms.ToTensor(),               # HWC → CHW, [0,255] → [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    elif name == "virchow2":
        
        model = encoder_factory(model_name="virchow2")
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    elif name == "ctranspath":
        
        model = encoder_factory(model_name="ctranspath")
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    elif name in ("hoptimus0", "hoptimus1"):
        
        model = encoder_factory(model_name=name)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])

    elif name in ("uni_v2", "uni_v1"):
        
        model = encoder_factory(model_name=name)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    elif name == "musk":
        
        model = encoder_factory(model_name="musk")
        transform = transforms.Compose([
            transforms.Resize(384, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN,
                std=IMAGENET_INCEPTION_STD
            ),
        ])

    elif name == "conch_v15":
        
        model = encoder_factory(model_name="conch_v15")
        transform = transforms.Compose([
            transforms.Resize(448, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    # ----- new backends -----
    # elif name == "mstar":
    #     # mSTAR from HuggingFace via timm
    #     model = timm.create_model(
    #         'hf-hub:Wangyh/mSTAR',
    #         pretrained=True,
    #         init_values=1e-5,
    #         dynamic_img_size=True
    #     )
    #     transform = transforms.Compose([
    #         transforms.Resize(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=(0.485, 0.456, 0.406),
    #             std=(0.229, 0.224, 0.225)
    #         ),
    #     ])

    # elif name in ("hibou-l", "hiboul"):
    #     # hibou-L from transformers
    #     processor = AutoImageProcessor.from_pretrained(
    #         "histai/hibou-L", trust_remote_code=True
    #     )
    #     model = AutoModel.from_pretrained(
    #         "histai/hibou-L", trust_remote_code=True
    #     )
    #     # transform returns the model-ready dict (pixel_values, etc.)
    #     transform = lambda img: processor(images=img, return_tensors="pt")

    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    return model, transform


# Example usage:
if __name__ == "__main__":
    for name in AVAILABLE_MODEL:
        model, tfm = get_model_and_transform(name)
        print(f"{name}: model={type(model)}  transform={tfm}")
