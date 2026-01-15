import argparse

parser = argparse.ArgumentParser(
    usage="python script.py models save_path"
)

parser.add_argument(
    "models",
    type=lambda s: s.split(","),
    help="Comma-separated list of model names (e.g. bert,resnet,vit)"
)

parser.add_argument(
    "save_path",
    type=str
)

args = parser.parse_args()

models = args.models      # ✅ list[str]
save_path = args.save_path

print(models)
print(save_path)