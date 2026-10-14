"""Evaluate one locked ParaMorph checkpoint once on the held-out test split."""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import LPBA40AtlasDataset, read_split
from metrics import assd_per_label, dice_per_label, nonpositive_jacobian_percentage, warp_labels
from models import ParaMorph, ParaMorphC
from train_paramorph import get_model_config, parse_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--atlas-image", required=True)
    parser.add_argument("--atlas-label", required=True)
    parser.add_argument("--test-split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=("paramorph", "paramorph-c", "paramorph-small"),
                        default="paramorph")
    parser.add_argument("--image-shape", type=parse_shape, default=(160, 192, 160))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device)
    model_tags = {
        "paramorph": "ParaMorph",
        "paramorph-c": "ParaMorph-C",
        "paramorph-small": "ParaMorph-S",
    }
    if state.get("architecture") != model_tags[args.model]:
        raise ValueError("Checkpoint architecture does not match --model")
    if tuple(state.get("image_shape", ())) != tuple(args.image_shape):
        raise ValueError("Checkpoint and requested image shapes differ")
    test_cases = read_split(args.test_split)
    recorded = state.get("splits", {}).get("test")
    if recorded is not None and list(recorded) != test_cases:
        raise ValueError("Test split differs from the split recorded during training")

    dataset = LPBA40AtlasDataset(
        test_cases, args.images_dir, args.atlas_image, args.image_shape,
        labels_dir=args.labels_dir, atlas_label=args.atlas_label, normalize=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model_classes = {
        "paramorph": ParaMorph,
        "paramorph-c": ParaMorphC,
        "paramorph-small": ParaMorph,
    }
    model = model_classes[args.model](get_model_config(args.model, args.image_shape)).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    spacing = tuple(float(v) for v in nib.load(args.atlas_image).header.get_zooms()[:3])

    rows = []
    with torch.no_grad():
        for batch in loader:
            moving = batch["moving"].to(device)
            fixed = batch["fixed"].to(device)
            _, flow = model(torch.cat((moving, fixed), dim=1))
            warped = warp_labels(batch["moving_label"].to(device), flow)
            prediction = warped[0, 0].cpu().numpy()
            target = batch["fixed_label"][0, 0].numpy()
            flow_np = flow[0].cpu().numpy()
            rows.append({
                "case_id": batch["case_id"][0],
                "dice": float(np.nanmean(dice_per_label(prediction, target))),
                "assd": float(np.nanmean(assd_per_label(prediction, target, spacing))),
                "nonpositive_jacobian_percentage": nonpositive_jacobian_percentage(flow_np),
            })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for metric in ("dice", "assd", "nonpositive_jacobian_percentage"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[metric] = {"mean": float(values.mean()),
                           "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    summary.update({"n": len(rows), "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_epoch": state["epoch"], "selection_metric": state["selection_metric"]})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
