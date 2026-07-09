import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageFile, PngImagePlugin

Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
PngImagePlugin.MAX_TEXT_MEMORY = 100 * 1024 * 1024
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


class PatchClassifier(nn.Module):
	def __init__(self, backbone: nn.Module, feature_dim: int, dropout: float):
		super().__init__()
		self.backbone = backbone
		self.classifier = nn.Sequential(
			nn.LayerNorm(feature_dim),
			nn.Dropout(dropout),
			nn.Linear(feature_dim, 2),
		)

	def forward(self, x):
		features = self.backbone(x)
		return self.classifier(features)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Fine-tune only the last transformer blocks of a patch encoder on positive/negative image folders."
	)
	parser.add_argument("data_dir", type=Path, help="Directory containing 'positive' and 'negative' folders")
	parser.add_argument("output_dir", type=Path, help="Directory to store checkpoints and training metadata")
	parser.add_argument(
		"--eval-data-dir",
		type=Path,
		help="Optional evaluation directory containing 'positive' and 'negative' folders",
	)
	parser.add_argument("--model", default="virchow2", choices=["virchow2", "conch_v15"], help="Backbone to fine-tune")
	parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
	parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
	parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
	parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for trainable parameters")
	parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
	parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout")
	parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--unfreeze-last-n-blocks", type=int, default=2, help="How many final transformer blocks to unfreeze")
	parser.add_argument("--attention-only", action="store_true", help="Only train attention submodules inside the unfrozen blocks")
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
	return parser.parse_args()


def set_seed(seed: int):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def validate_data_dir(data_dir: Path):
	expected_dirs = [data_dir / "positive", data_dir / "negative"]
	missing = [str(path) for path in expected_dirs if not path.is_dir()]
	if missing:
		raise FileNotFoundError(
			"Expected ImageFolder layout with class folders 'positive' and 'negative'. Missing: "
			+ ", ".join(missing)
		)


def stratified_split(dataset, val_split: float, seed: int):
	if not 0.0 < val_split < 1.0:
		raise ValueError("val_split must be between 0 and 1")

	targets = dataset.targets
	train_indices = []
	val_indices = []
	rng = random.Random(seed)

	for class_id in sorted(set(targets)):
		indices = [index for index, target in enumerate(targets) if target == class_id]
		rng.shuffle(indices)
		val_count = max(1, int(len(indices) * val_split))
		if len(indices) - val_count < 1:
			val_count = len(indices) - 1
		if val_count < 1:
			raise ValueError("Each class needs at least 2 samples for train/validation split")
		val_indices.extend(indices[:val_count])
		train_indices.extend(indices[val_count:])

	rng.shuffle(train_indices)
	rng.shuffle(val_indices)
	return train_indices, val_indices


def get_transformer_blocks(backbone: nn.Module):
	if hasattr(backbone.model, "blocks"):
		return backbone.model.blocks
	if hasattr(backbone.model, "trunk") and hasattr(backbone.model.trunk, "blocks"):
		return backbone.model.trunk.blocks
	raise ValueError("Backbone does not expose transformer blocks for selective fine-tuning")


def configure_trainable_layers(backbone: nn.Module, unfreeze_last_n_blocks: int, attention_only: bool):
	for parameter in backbone.parameters():
		parameter.requires_grad = False

	blocks = get_transformer_blocks(backbone)
	if unfreeze_last_n_blocks < 1 or unfreeze_last_n_blocks > len(blocks):
		raise ValueError(f"unfreeze_last_n_blocks must be in [1, {len(blocks)}]")

	trainable_block_count = 0
	for block in blocks[-unfreeze_last_n_blocks:]:
		trainable_block_count += 1
		if attention_only:
			for _, parameter in block.attn.named_parameters():
				parameter.requires_grad = True
		else:
			for parameter in block.parameters():
				parameter.requires_grad = True

	for name, parameter in backbone.named_parameters():
		if name.endswith("norm.weight") or name.endswith("norm.bias"):
			parameter.requires_grad = True

	return trainable_block_count


def build_dataloaders(
	data_dir: Path,
	eval_data_dir: Path | None,
	transform,
	batch_size: int,
	num_workers: int,
	val_split: float,
	seed: int,
):
	train_dataset = datasets.ImageFolder(data_dir, transform=transform)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)

	if eval_data_dir is None:
		train_indices, val_indices = stratified_split(train_dataset, val_split=val_split, seed=seed)
		train_dataset = Subset(train_dataset, train_indices)
		val_dataset = Subset(datasets.ImageFolder(data_dir, transform=transform), val_indices)
		train_loader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=num_workers,
			pin_memory=torch.cuda.is_available(),
		)
	else:
		val_dataset = datasets.ImageFolder(eval_data_dir, transform=transform)
		if train_dataset.class_to_idx != val_dataset.class_to_idx:
			raise ValueError(
				"Training and evaluation directories must expose the same class folders. "
				f"Train classes: {train_dataset.class_to_idx}, eval classes: {val_dataset.class_to_idx}"
			)

	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	return train_dataset, val_dataset, train_loader, val_loader


def infer_feature_dim(backbone: nn.Module, dataset, device: torch.device):
	sample, _ = dataset[0]
	sample = sample.unsqueeze(0).to(device)
	with torch.no_grad():
		features = backbone(sample)
	return features.shape[-1]


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
	model.eval()
	total_loss = 0.0
	total_correct = 0
	total_examples = 0

	with torch.no_grad():
		for images, labels in dataloader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			logits = model(images)
			loss = criterion(logits, labels)
			total_loss += loss.item() * labels.size(0)
			total_correct += (logits.argmax(dim=1) == labels).sum().item()
			total_examples += labels.size(0)

	return total_loss / total_examples, total_correct / total_examples


def cpu_state_dict(module: nn.Module):
	return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def main():
	args = parse_args()
	from helpers import get_model_and_transform

	set_seed(args.seed)
	validate_data_dir(args.data_dir)
	if args.eval_data_dir is not None:
		validate_data_dir(args.eval_data_dir)
	args.output_dir.mkdir(parents=True, exist_ok=True)

	device = torch.device(args.device)
	backbone, transform = get_model_and_transform(args.model)
	backbone.to(device)

	train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
		data_dir=args.data_dir,
		eval_data_dir=args.eval_data_dir,
		transform=transform,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		val_split=args.val_split,
		seed=args.seed,
	)

	feature_dim = infer_feature_dim(backbone, train_dataset, device)
	unfrozen_blocks = configure_trainable_layers(
		backbone=backbone,
		unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
		attention_only=args.attention_only,
	)

	model = PatchClassifier(backbone=backbone, feature_dim=feature_dim, dropout=args.dropout).to(device)

	for parameter in model.classifier.parameters():
		parameter.requires_grad = True

	trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
	if not trainable_params:
		raise RuntimeError("No trainable parameters were enabled")

	train_targets = train_dataset.targets if hasattr(train_dataset, "targets") else [train_dataset.dataset.targets[i] for i in train_dataset.indices]
	class_counts = torch.bincount(torch.tensor(train_targets), minlength=len(train_dataset.classes)).float()
	class_weights = class_counts.sum() / class_counts.clamp_min(1.0)
	class_weights = (class_weights / class_weights.mean()).to(device)

	optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
	criterion = nn.CrossEntropyLoss(weight=class_weights)

	best_state = None
	best_val_accuracy = -1.0
	history = []
	eval_source = str(args.eval_data_dir) if args.eval_data_dir is not None else "split-from-train"

	print(f"Using model: {args.model}")
	print(f"Classes: {train_dataset.class_to_idx}")
	print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
	print(f"Evaluation source: {eval_source}")
	print(f"Unfrozen transformer blocks: {unfrozen_blocks}, attention_only={args.attention_only}")
	print(f"Trainable parameter tensors: {sum(1 for parameter in trainable_params)}")

	for epoch in range(1, args.epochs + 1):
		model.train()
		epoch_loss = 0.0
		epoch_correct = 0
		epoch_examples = 0

		for images, labels in train_loader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)
			logits = model(images)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item() * labels.size(0)
			epoch_correct += (logits.argmax(dim=1) == labels).sum().item()
			epoch_examples += labels.size(0)

		train_loss = epoch_loss / epoch_examples
		train_accuracy = epoch_correct / epoch_examples
		val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

		epoch_summary = {
			"epoch": epoch,
			"train_loss": train_loss,
			"train_accuracy": train_accuracy,
			"val_loss": val_loss,
			"val_accuracy": val_accuracy,
		}
		history.append(epoch_summary)
		print(json.dumps(epoch_summary))

		if val_accuracy > best_val_accuracy:
			best_val_accuracy = val_accuracy
			best_state = cpu_state_dict(model)
			torch.save(cpu_state_dict(backbone.model), args.output_dir / f"{args.model}_backbone_best.pt")

	if best_state is None:
		raise RuntimeError("Training finished without producing a checkpoint")

	checkpoint = {
		"model_name": args.model,
		"class_to_idx": train_dataset.class_to_idx,
		"args": vars(args),
		"history": history,
		"best_val_accuracy": best_val_accuracy,
		"model_state_dict": best_state,
	}
	torch.save(checkpoint, args.output_dir / f"{args.model}_finetune.pt")

	metadata = {
		"model_name": args.model,
		"class_to_idx": train_dataset.class_to_idx,
		"best_val_accuracy": best_val_accuracy,
		"eval_data_dir": str(args.eval_data_dir) if args.eval_data_dir is not None else None,
		"output_backbone": str(args.output_dir / f"{args.model}_backbone_best.pt"),
		"output_checkpoint": str(args.output_dir / f"{args.model}_finetune.pt"),
	}
	with open(args.output_dir / f"{args.model}_finetune_metrics.json", "w", encoding="utf-8") as handle:
		json.dump({"history": history, "metadata": metadata}, handle, indent=2)


if __name__ == "__main__":
	main()