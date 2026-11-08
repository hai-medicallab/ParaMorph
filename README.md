# ParaMorph

**Registration-Oriented Local-Global Deformation Modeling for Unsupervised
Volumetric Medical Image Registration**

ParaMorph is an unsupervised 3-D deformable image registration network that
combines a hierarchical ConvNet pathway with a hierarchical Transformer
pathway. The ConvNet pathway preserves local anatomical and boundary detail,
while the Transformer pathway models long-range spatial correspondence. Their
features are aggregated at multiple resolutions and decoded into a dense
displacement field that warps a moving volume into a fixed-atlas space.

This repository contains the LPBA40 reference implementation used in the
revised manuscript. It provides the ParaMorph, ParaMorph-C, and ParaMorph-S
architectures, fixed LPBA40 split manifests, validation-based checkpoint
selection, a separate held-out test entry point, and the evaluation metrics
used for LPBA40. HPH-PM and Prostate158 data and execution packages are not
included.

## Highlights

- Parallel 3-D ConvNet and Swin Transformer feature hierarchies
- Multi-resolution aggregation of local and global deformation cues
- ConvNeXt-style refinement of shallow skip features before decoding
- A consistent moving-to-fixed deformation convention for images and labels
- Fixed and mutually exclusive LPBA40 training, validation, and test splits
- Separate model selection and held-out evaluation entry points
- Case-level DSC, ASSD, and non-positive Jacobian percentage reporting
- Explicit per-label audit of empty masks excluded from ASSD
- Reproducible three-seed paired statistical analysis for manuscript Table 3
- A declarative manifest for the component-isolation rows in Tables 6-7
- ParaMorph-S for an approximately parameter-matched comparison with
  TransMorph

## Model variants

| Model | Description | Trainable parameters |
|---|---|---:|
| ParaMorph | Full parallel ConvNet-Transformer model | 63,328,195 |
| ParaMorph-C | ParaMorph with parameter-free cross-pathway interaction | 63,328,195 |
| ParaMorph-S | Full ParaMorph organization with base width reduced from 96 to 84 | 48,569,521 |

ParaMorph-C adds channel-aligned ConvNet tokens to the corresponding
Transformer-stage outputs by tensor reshaping and element-wise addition. It
therefore has the same number of trainable parameters as ParaMorph.
ParaMorph-S retains the complete dual-pathway, aggregation, skip-refinement,
and decoder organization while reducing the shared width. Its parameter count
is within 3.90% of the 46.746-million-parameter TransMorph comparator reported
in the manuscript.

## Registration pipeline

1. The moving and fixed images are concatenated along the channel dimension.
2. The Transformer pathway extracts long-range spatial representations.
3. The ConvNet pathway extracts hierarchical local anatomical features.
4. Features from the two pathways are aligned and aggregated at four spatial
   resolutions.
5. Refined skip features and aggregated deep features are decoded into a dense
   displacement field.
6. The moving image is resampled into the unchanged fixed-atlas grid with
   trilinear interpolation. Moving labels are propagated with nearest-neighbor
   interpolation during evaluation.

The fixed image and fixed label are never warped during evaluation.

## Release scope and dataset protocol

The manuscript evaluates ParaMorph separately on three subject-to-atlas MRI
datasets. One prespecified case is used as the unchanged fixed atlas for each
dataset and is not counted among the moving-image split sizes.

| Dataset | Fixed atlas | Training | Validation | Held-out test | Model input |
|---|---:|---:|---:|---:|---:|
| HPH-PM | HPH-PM-001 | 90 | 10 | 10 | 160 x 160 x 96 |
| Prostate158 | t2_001 | 129 | 10 | 18 | 160 x 160 x 96 |
| LPBA40 | S01 | 25 | 5 | 9 | 160 x 192 x 160 |

Only the LPBA40 pipeline is included in this repository. The released loader
expects the prepared LPBA40 arrays to retain their supplied `160 x 192 x 160`
matrix and isotropic `1 mm` spacing. It checks the array shape and does not
silently crop, pad, or resize the input.

## LPBA40 splits

The split manifests under `splits/` define the complete moving-image protocol:

- Training: S11-S35 (`lpba40_train.txt`)
- Validation: S36-S40 (`lpba40_validation.txt`)
- Held-out test: S02-S10 (`lpba40_test.txt`)
- Fixed atlas: S01

The training program verifies that the three moving-image subsets are
disjoint. Test identifiers are read during training only for this overlap
check; test images and labels are not opened by the training program.

## Expected LPBA40 layout

```text
LPBA40/
  fixed.nii.gz
  train/
    S11.delineation.skullstripped.nii.gz
    ...
    S40.delineation.skullstripped.nii.gz
  test/
    S02.delineation.skullstripped.nii.gz
    ...
    S10.delineation.skullstripped.nii.gz
  label/
    S01.delineation.structure.label.nii.gz
    S02.delineation.skullstripped.nii.gz
    ...
    S10.delineation.skullstripped.nii.gz
    S11.delineation.structure.label.nii.gz
    ...
    S40.delineation.structure.label.nii.gz
```

Prepared LPBA40 copies use two label filename conventions. The loader first
checks `SXX.delineation.structure.label.nii.gz` and then checks
`SXX.delineation.skullstripped.nii.gz` in the label directory. This fallback
only resolves filenames; it does not modify image or label values.

## Installation

Python 3.8 or later is recommended. Install a CUDA-compatible PyTorch build if
GPU execution is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Training

The released LPBA40 configuration uses 500 epochs, batch size 1, Adam,
polynomial learning-rate decay, voxelwise mean-squared error, and a
first-order displacement smoothness weight of `0.02`. Validation DSC alone is
used to select `best.pt`.

```bash
python train_paramorph.py \
  --images-dir /path/LPBA40/train \
  --labels-dir /path/LPBA40/label \
  --atlas-image /path/LPBA40/fixed.nii.gz \
  --atlas-label /path/LPBA40/label/S01.delineation.structure.label.nii.gz \
  --train-split splits/lpba40_train.txt \
  --val-split splits/lpba40_validation.txt \
  --test-split splits/lpba40_test.txt \
  --output-dir runs/lpba40_seed47 \
  --seed 47 \
  --model paramorph
```

Select another released architecture with:

- `--model paramorph-c` for ParaMorph-C
- `--model paramorph-small` for ParaMorph-S

The manuscript reports ParaMorph-S with seed 47 only. TransMorph and full
ParaMorph were trained independently with seeds 47, 726, and 2026 for the
repeated-training analysis.

### Optional augmentation

Augmentation is disabled by default and was not used for the reported LPBA40
experiments. Synchronized random flips and small-angle rotations are available
for optional experiments:

```bash
--augment --rotation-degrees 5 --rotation-probability 0.5
```

The sampled transform is shared by the moving and fixed images. Validation and
test data are never augmented. Scaling, elastic deformation, gamma
augmentation, and intensity perturbation are not implemented or claimed.

## Held-out evaluation

After model selection is complete, evaluate the validation-selected checkpoint
on the held-out test split:

```bash
python test_paramorph.py \
  --images-dir /path/LPBA40/test \
  --labels-dir /path/LPBA40/label \
  --atlas-image /path/LPBA40/fixed.nii.gz \
  --atlas-label /path/LPBA40/label/S01.delineation.structure.label.nii.gz \
  --test-split splits/lpba40_test.txt \
  --checkpoint runs/lpba40_seed47/best.pt \
  --output-dir results/lpba40_seed47 \
  --model paramorph
```

The evaluation program writes:

- `per_case_metrics.csv`: DSC, ASSD, and folding percentage for every test case,
  formatted to exactly three digits after the decimal point
- `empty_label_audit.csv`: every label excluded from ASSD because the warped or
  fixed-reference mask was empty, including the case ID and the empty side
- `summary.json`: aggregate means and sample standard deviations rounded to
  three digits after the decimal point

The checkpoint architecture is checked against `--model` before inference.

## Evaluation metrics

- **DSC:** calculated for each foreground label and macro-averaged within each
  registration pair.
- **ASSD:** the mean of the two directed surface distances. LPBA40 uses
  `(1, 1, 1)` spacing, so the reported values are in millimetres. If either
  the warped mask or reference mask is empty for an evaluated label, that
  label returns `NaN` and is excluded from the within-case macro average. The
  rule is never applied silently: all exclusions are written to
  `empty_label_audit.csv`, and their number is recorded in `summary.json`.
- **Folding percentage:**
  `100 x mean(det(J_phi) <= 0)`, where `J_phi` is the spatial Jacobian of the
  identity-plus-displacement transformation.

LPBA40 evaluation uses the 54 foreground IDs listed in `metrics.py`. Raw label
IDs 181 and 182 are excluded to match the evaluation protocol described in the
manuscript.

## Table 3 paired statistics

`statistics/table3_paired_analysis.py` reproduces the manuscript's paired
analysis from the bundled anonymized case-level three-seed means. It applies
two-sided Wilcoxon signed-rank tests, calculates rank-biserial effect sizes,
obtains paired BCa bootstrap confidence intervals for the mean improvement,
and applies a joint Holm correction to the nine comparisons.

```bash
python statistics/table3_paired_analysis.py \
  --case-file statistics/table3_case_level_three_seed_means.csv \
  --output results/table3_paired_statistics.csv
```

The repository includes the script, the anonymized three-decimal case-level
input, and the expected aggregate Table 3 output. The script also retains the
`--manifest` option for recomputation from separate per-seed evaluator
outputs; see `statistics/README.md` and `statistics/example_manifest.csv`.

## Ablation configuration map

`configs/ablation_manifest.json` maps every component-isolation row in Tables
6-7 to its architecture switches and, where available, its exact preserved
model and training source under `legacy_ablations/`. In particular,
`combined_encoders_no_gf_pmfe` uses parameter-free element-wise addition at
four matched resolutions. `baseline_plus_gf` refines the three Transformer
skip features separately before decoding while retaining the deepest
Transformer feature and the two original shallow TransMorph convolutional
skips. See `legacy_ablations/README.md` for the row-to-source map.

## Repository structure

```text
Paramorph/
  data/
    datasets.py
  models/
    configs.py
    convnet_only.py
    paramorph.py
    paramorph_c.py
  splits/
    lpba40_train.txt
    lpba40_validation.txt
    lpba40_test.txt
  configs/
    ablation_manifest.json
  legacy_ablations/
    README.md
    models/
    train_TransMorph_*.py
  statistics/
    table3_paired_analysis.py
    example_manifest.csv
    table3_case_level_three_seed_means.csv
    table3_expected_output.csv
    README.md
  metrics.py
  train_paramorph.py
  test_paramorph.py
  requirements.txt
  README.md
```

Trained checkpoints, experiment logs, generated caches, and datasets are not
included in the repository.

## Data availability

LPBA40 is available from its original public source. Users should obtain the
dataset under its applicable terms and prepare the directory structure shown
above. The in-house HPH-PM dataset is not publicly distributed, and this code
release does not include HPH-PM or Prostate158 execution packages.

## Citation

The citation will be added after publication. If this code supports your work,
please cite the ParaMorph article once its final bibliographic information is
available.
