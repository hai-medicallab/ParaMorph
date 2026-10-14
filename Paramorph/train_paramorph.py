"""Train ParaMorph using validation-only checkpoint selection.

The held-out test split is read only as identifiers for a disjointness audit.
No test image, label, or metric is accessed by this script.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.datasets import LPBA40AtlasDataset, assert_disjoint_splits, read_split
from metrics import dice_per_label, warp_labels
from models import (
    ParaMorph, ParaMorphC, get_paramorph_config, get_paramorph_small_config,
)


MODEL_CLASSES = {
    "paramorph": ParaMorph,
    "paramorph-c": ParaMorphC,
    "paramorph-small": ParaMorph,
}
MODEL_TAGS = {
    "paramorph": "ParaMorph",
    "paramorph-c": "ParaMorph-C",
    "paramorph-small": "ParaMorph-S",
}


def get_model_config(model_name, image_shape):
    if model_name == "paramorph-small":
        return get_paramorph_small_config(image_shape)
    return get_paramorph_config(image_shape)


def parse_shape(text):
    value = tuple(int(item) for item in text.split(","))
    if len(value) != 3:
        raise argparse.ArgumentTypeError("shape must be D,H,W")
    return value


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def smoothness_loss(flow):
    differences = [flow[:, :, 1:] - flow[:, :, :-1],
                   flow[:, :, :, 1:] - flow[:, :, :, :-1],
                   flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]]
    return sum(torch.mean(delta.square()) for delta in differences) / 3.0


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    case_scores = []
    for batch in loader:
        moving = batch["moving"].to(device, non_blocking=True)
        fixed = batch["fixed"].to(device, non_blocking=True)
        _, flow = model(torch.cat((moving, fixed), dim=1))
        warped = warp_labels(batch["moving_label"].to(device), flow)
        pred = warped[0, 0].cpu().numpy()
        target = batch["fixed_label"][0, 0].numpy()
        case_scores.append(float(np.nanmean(dice_per_label(pred, target))))
    return float(np.mean(case_scores))


def save_checkpoint(path, model, optimizer, epoch, best_val_dice, args, splits):
    torch.save({
        "architecture": MODEL_TAGS[args.model],
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_validation_dice": best_val_dice,
        "image_shape": args.image_shape,
        "seed": args.seed,
        "splits": splits,
        "selection_metric": "mean validation DSC",
        "flow_convention": "moving_to_fixed",
        "augmentation": {
            "enabled": args.augment,
            "synchronized_random_flip_per_axis_probability": 0.5 if args.augment else 0.0,
            "rotation_degrees": args.rotation_degrees if args.augment else 0.0,
            "rotation_probability": args.rotation_probability if args.augment else 0.0,
            "spatial_transform_shared_within_pair": True,
        },
    }, path)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--atlas-image", required=True)
    parser.add_argument("--atlas-label", required=True)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--test-split", required=True,
                        help="Identifiers are used only to verify split disjointness")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_CLASSES), default="paramorph")
    parser.add_argument("--image-shape", type=parse_shape, default=(160, 192, 160))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument(
        "--augment", action="store_true",
        help="Optionally enable synchronized flips and small-angle rotations",
    )
    parser.add_argument("--rotation-degrees", type=float, default=5.0)
    parser.add_argument("--rotation-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--max-train-cases", type=int)
    parser.add_argument("--max-val-cases", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cases = read_split(args.train_split)
    val_cases = read_split(args.val_split)
    test_cases = read_split(args.test_split)
    splits = {"train": train_cases, "validation": val_cases, "test": test_cases}
    assert_disjoint_splits(**splits)
    if args.max_train_cases:
        train_cases = train_cases[:args.max_train_cases]
    if args.max_val_cases:
        val_cases = val_cases[:args.max_val_cases]

    train_set = LPBA40AtlasDataset(
        train_cases, args.images_dir, args.atlas_image, args.image_shape,
        normalize=True, random_flip=args.augment,
        rotation_degrees=args.rotation_degrees,
        rotation_probability=args.rotation_probability if args.augment else 0.0,
    )
    val_set = LPBA40AtlasDataset(
        val_cases, args.images_dir, args.atlas_image, args.image_shape,
        labels_dir=args.labels_dir, atlas_label=args.atlas_label, normalize=True,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", generator=generator,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers)

    config = get_model_config(args.model, args.image_shape)
    model = MODEL_CLASSES[args.model](config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    start_epoch, best_val_dice = 0, -np.inf
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        if state.get("architecture") != MODEL_TAGS[args.model]:
            raise ValueError("Checkpoint architecture does not match --model")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        best_val_dice = float(state["best_validation_dice"])

    history_path = output_dir / "training_history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        lr = args.learning_rate * (1.0 - epoch / args.epochs) ** 0.9
        for group in optimizer.param_groups:
            group["lr"] = lr
        for step, batch in enumerate(train_loader):
            moving = batch["moving"].to(device, non_blocking=True)
            fixed = batch["fixed"].to(device, non_blocking=True)
            warped, flow = model(torch.cat((moving, fixed), dim=1))
            similarity = F.mse_loss(warped, fixed)
            regularity = smoothness_loss(flow)
            loss = similarity + args.smoothness_weight * regularity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if args.max_steps_per_epoch and step + 1 >= args.max_steps_per_epoch:
                break

        val_dice = validate(model, val_loader, device)
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                  "validation_dice": val_dice, "learning_rate": lr}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch + 1,
                            best_val_dice, args, splits)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch + 1,
                        best_val_dice, args, splits)


if __name__ == "__main__":
    main()
