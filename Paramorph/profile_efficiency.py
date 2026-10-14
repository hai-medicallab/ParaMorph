"""Profile model complexity and CUDA inference efficiency reproducibly.

The default protocol matches the manuscript efficiency table: one 3-D image
pair, batch size 1, and a 160 x 160 x 96 input grid. THOP reports MACs; the
script reports GFLOPs as 2 * MACs, matching the manuscript convention.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from thop import profile

MODEL_NAMES = ("transmorph", "paramorph-small", "paramorph", "paramorph-c")


def parse_shape(text):
    shape = tuple(int(value) for value in text.split(","))
    if len(shape) != 3:
        raise argparse.ArgumentTypeError("shape must be D,H,W")
    return shape


def build_model(name, shape):
    if name == "transmorph":
        # Load the retained TransMorph reference implementation only for the
        # capacity-matched resource comparison.
        import sys

        baseline_root = (
            Path(__file__).resolve().parent.parent
            / "代码" / "TransMorph-LBPA40" / "TransMorph-LBPA40"
        )
        if not baseline_root.exists():
            raise FileNotFoundError(f"TransMorph reference code not found: {baseline_root}")
        sys.path.insert(0, str(baseline_root))
        from models.TransMorph import TransMorph
        from models.configs_TransMorph import get_3DTransMorph_config

        config = get_3DTransMorph_config()
        config.img_size = tuple(shape)
        return TransMorph(config)

    from models import (
        ParaMorph, ParaMorphC, get_paramorph_config,
        get_paramorph_small_config,
    )

    if name == "paramorph-small":
        config = get_paramorph_small_config(shape)
    else:
        config = get_paramorph_config(shape)
    model_class = ParaMorphC if name == "paramorph-c" else ParaMorph
    return model_class(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--shape", type=parse_shape, default=(160, 160, 96))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--training-memory", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory and timing measurements")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model = build_model(args.model, args.shape)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = dict(state["state_dict"])
        # The sampling grid is a deterministic shape-specific buffer rather
        # than a learned weight. Recreate it for the profiling input grid.
        state_dict.pop("spatial_trans.grid", None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing != ["spatial_trans.grid"] or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    model.eval().to(device)
    sample = torch.randn((1, 2, *args.shape), device=device)

    with torch.inference_mode():
        macs, thop_params = profile(model, inputs=(sample,), verbose=False)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(sample)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    timings_ms = []
    with torch.inference_mode():
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(sample)
            end.record()
            torch.cuda.synchronize(device)
            timings_ms.append(start.elapsed_time(end))

    result = {
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "input_shape": [1, 2, *args.shape],
        "trainable_parameters": trainable_params,
        "trainable_parameters_millions": trainable_params / 1e6,
        "thop_parameters": int(thop_params),
        "macs_giga": macs / 1e9,
        "gflops_two_per_mac": 2.0 * macs / 1e9,
        "peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "inference_time_mean_seconds": statistics.mean(timings_ms) / 1000.0,
        "inference_time_std_seconds": statistics.stdev(timings_ms) / 1000.0 if len(timings_ms) > 1 else 0.0,
        "inference_time_median_seconds": statistics.median(timings_ms) / 1000.0,
        "measured_at_unix_time": time.time(),
    }
    if args.training_memory:
        # Match the deterministic training configuration used by train_paramorph.py.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        moved, flow = model(sample)
        loss = moved.square().mean() + flow.square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        result["training_step_peak_memory_allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024**2
        )
        result["training_step_peak_memory_reserved_mb"] = (
            torch.cuda.max_memory_reserved(device) / 1024**2
        )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
