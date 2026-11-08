# Table 3 paired analysis

`table3_paired_analysis.py` implements the statistical procedure described in
the manuscript. The anonymized file
`table3_case_level_three_seed_means.csv` contains the three-seed mean for
TransMorph and ParaMorph for every held-out registration unit used in Table 3.
All three metrics are stored with exactly three digits after the decimal point.

Run the analysis directly from the bundled case-level data:

```bash
python statistics/table3_paired_analysis.py \
  --case-file statistics/table3_case_level_three_seed_means.csv \
  --output statistics/table3_results.csv
```

Case identifiers are anonymized and are used only to preserve the pairing
between TransMorph and ParaMorph. Folding values are percentages.

The script also supports recomputation from separate per-seed evaluator
outputs. Create a manifest with one row per dataset, method, and seed. The
required columns are `dataset`, `method`, `seed`, and `path`. Method names must
be `TransMorph` and `ParaMorph`. `example_manifest.csv` shows the format.

To recompute from the separate per-seed files, run:

```bash
python statistics/table3_paired_analysis.py \
  --manifest statistics/example_manifest.csv \
  --output statistics/table3_results.csv
```

When a manifest is used, the script first averages DSC, ASSD, and folding
percentage across seeds 47, 726, and 2026 for each method and case. It then calculates the signed
paired improvement (ParaMorph minus TransMorph for DSC; TransMorph minus
ParaMorph for ASSD and folding), a two-sided Wilcoxon signed-rank test, a
paired BCa bootstrap 95% confidence interval for the mean improvement using
100,000 resamples and random seed 47, rank-biserial correlation, and one Holm
adjustment across the nine comparisons. `table3_expected_output.csv` records
the aggregate values reported in the manuscript for comparison.
