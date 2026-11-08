"""Reproduce the paired statistical procedure reported in manuscript Table 3.

The script accepts either a manifest pointing to per-seed ``per_case_metrics.csv``
files or the bundled anonymized case-level file containing the three-seed mean
for each method and registration unit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, rankdata, wilcoxon


METRICS = (
    ("DSC", "dice", 1.0),
    ("ASSD", "assd", -1.0),
    ("Folding", "nonpositive_jacobian_percentage", -1.0),
)


def read_manifest(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            csv_path = Path(item["path"])
            if not csv_path.is_absolute():
                csv_path = (path.parent / csv_path).resolve()
            rows.append({**item, "path": csv_path, "seed": int(item["seed"])})
    if not rows:
        raise ValueError("The manifest contains no evaluation files")
    return rows


def load_case_rows(manifest_rows):
    records = []
    for entry in manifest_rows:
        with entry["path"].open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                records.append({
                    "dataset": entry["dataset"],
                    "method": entry["method"],
                    "seed": entry["seed"],
                    "case_id": row["case_id"],
                    "dice": float(row["dice"]),
                    "assd": float(row["assd"]),
                    "nonpositive_jacobian_percentage": float(
                        row["nonpositive_jacobian_percentage"]
                    ),
                })
    return records


def read_case_file(path: Path):
    records = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            records.append({
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": row["seed"],
                "case_id": row["case_id"],
                "dice": float(row["dice"]),
                "assd": float(row["assd"]),
                "nonpositive_jacobian_percentage": float(
                    row["nonpositive_jacobian_percentage"]
                ),
            })
    if not records:
        raise ValueError("The case-level metric file contains no records")
    return records


def seed_average(records, dataset, method, column):
    values = {}
    for row in records:
        if row["dataset"] == dataset and row["method"] == method:
            values.setdefault(row["case_id"], []).append((row["seed"], row[column]))
    averaged = {}
    for case_id, pairs in values.items():
        seeds = [seed for seed, _ in pairs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed for {dataset}/{method}/{case_id}")
        averaged[case_id] = float(np.mean([value for _, value in pairs]))
    return averaged


def rank_biserial(differences):
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    return float(np.sum(np.sign(nonzero) * ranks) / np.sum(ranks))


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for position, index in enumerate(order):
        running = max(running, (m - position) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def analyse(records, n_resamples, random_seed):
    datasets = sorted({row["dataset"] for row in records})
    results = []
    for dataset in datasets:
        for metric_name, column, direction in METRICS:
            trans = seed_average(records, dataset, "TransMorph", column)
            para = seed_average(records, dataset, "ParaMorph", column)
            case_ids = sorted(set(trans) & set(para))
            if set(trans) != set(para):
                raise ValueError(f"Unpaired cases for {dataset}/{metric_name}")
            if not case_ids:
                raise ValueError(f"No paired cases for {dataset}/{metric_name}")
            differences = direction * np.asarray(
                [para[case_id] - trans[case_id] for case_id in case_ids], dtype=float
            )
            test = wilcoxon(differences, alternative="two-sided", method="auto")
            interval = bootstrap(
                (differences,),
                np.mean,
                vectorized=False,
                paired=True,
                confidence_level=0.95,
                n_resamples=n_resamples,
                method="BCa",
                random_state=random_seed,
            ).confidence_interval
            results.append({
                "dataset": dataset,
                "metric": metric_name,
                "n_pairs": len(case_ids),
                "mean_improvement": float(np.mean(differences)),
                "ci_low": float(interval.low),
                "ci_high": float(interval.high),
                "rank_biserial": rank_biserial(differences),
                "wilcoxon_W": float(test.statistic),
                "p_unadjusted": float(test.pvalue),
            })
    adjusted = holm_adjust([row["p_unadjusted"] for row in results])
    for row, value in zip(results, adjusted):
        row["p_holm"] = float(value)
    return results


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--case-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=47)
    args = parser.parse_args()

    records = (
        read_case_file(args.case_file)
        if args.case_file is not None
        else load_case_rows(read_manifest(args.manifest))
    )
    results = analyse(records, args.bootstrap_resamples, args.random_seed)
    output_rows = []
    for row in results:
        output_rows.append({
            "dataset": row["dataset"],
            "metric": row["metric"],
            "n_pairs": str(row["n_pairs"]),
            "mean_improvement": f'{row["mean_improvement"]:.3f}',
            "ci_low": f'{row["ci_low"]:.3f}',
            "ci_high": f'{row["ci_high"]:.3f}',
            "rank_biserial": f'{row["rank_biserial"]:.3f}',
            "wilcoxon_W": (
                str(int(row["wilcoxon_W"]))
                if row["wilcoxon_W"].is_integer()
                else f'{row["wilcoxon_W"]:.1f}'
            ),
            "p_unadjusted": f'{row["p_unadjusted"]:.3f}',
            "p_holm": f'{row["p_holm"]:.3f}',
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(results)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
