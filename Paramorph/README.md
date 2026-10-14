# ParaMorph

**Registration-Oriented Local-Global Deformation Modeling for Unsupervised
Volumetric Medical Image Registration**

ParaMorph is an unsupervised 3-D medical image registration network with
parallel ConvNet and Transformer feature hierarchies. The ConvNet pathway
preserves local anatomical detail, while the Transformer pathway models
long-range spatial correspondence. Multi-resolution features from both
pathways are aggregated to estimate a dense displacement field.

This public release is intentionally scoped to LPBA40. It contains ParaMorph,
ParaMorph-C, and the capacity-matched ParaMorph-S; fixed data splits; separate
training and held-out testing scripts; metric functions; and reproducible
commands. Trained checkpoints and complete HPH-PM or Prostate158 execution
packages are not included.

## Method overview

ParaMorph contains three main parts:

1. **Parallel encoders.** A hierarchical 3-D Swin Transformer pathway captures
   broad spatial correspondence, while a ConvNet pathway based on enhanced
   residual ConvNeXt-style blocks preserves local boundary information.
2. **Multi-resolution deformation-cue aggregation.** Channel-aligned features
   from the two pathways are aggregated at four corresponding resolutions.
3. **Skip-feature refinement and deformation decoding.** PMFE modules refine
   shallow skip features before they enter the convolutional decoder, which
   predicts the dense displacement field.

ParaMorph-C additionally adds channel-aligned ConvNet tokens to corresponding
Transformer-stage outputs before normalization. This interaction uses tensor
reshaping and element-wise addition, so ParaMorph and ParaMorph-C have the same
number of trainable parameters.

ParaMorph-S retains the full ParaMorph structure but reduces the shared base
width from 96 to 84 channels. It contains 48,569,521 trainable parameters,
within 3.90% of the 46.746-million-parameter TransMorph comparator. Full
ParaMorph and ParaMorph-C each contain 63,328,195 trainable parameters.

## Datasets described in the manuscript

The manuscript evaluates each model separately on three subject-to-atlas MRI
datasets. The fixed atlas is not included in the moving-image split counts.

| Dataset | Fixed atlas | Training | Validation | Held-out test | Input size |
|---|---:|---:|---:|---:|---:|
| HPH-PM | HPH-PM-001 | 90 | 10 | 10 | 160 x 160 x 96 |
| Prostate158 | t2_001 | 129 (t2_020-t2_148) | 10 (t2_149-t2_158) | 18 (t2_002-t2_019) | 160 x 160 x 96 |
| LPBA40 | S01 | 25 (S11-S35) | 5 (S36-S40) | 9 (S02-S10) | 160 x 192 x 160 |

The released code covers LPBA40 only. LPBA40 images are expected to retain the
supplied `160 x 192 x 160` matrix and isotropic `1 mm` spacing. The loader
checks this shape and does not silently crop, pad, or resize data.

## Registration convention

Each participant is the **moving** image and S01 is the unchanged **fixed**
atlas. The network receives the channel-wise concatenation of moving and fixed
images. The predicted field resamples the moving image into the fixed-atlas
grid. Moving labels use nearest-neighbor interpolation. The fixed image and
fixed label are not warped during evaluation.

## LPBA40 layout

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

The label loader supports both label filename conventions found in prepared
LPBA40 copies. It first checks
`SXX.delineation.structure.label.nii.gz` and, if absent, checks
`SXX.delineation.skullstripped.nii.gz` in the label directory. This fallback
changes path resolution only and does not modify image or label data.

The fixed split manifests are located in `splits/`:

- `lpba40_train.txt`: S11-S35
- `lpba40_validation.txt`: S36-S40
- `lpba40_test.txt`: S02-S10

The training script checks that all three moving-image subsets are disjoint.
Test identifiers are read during training only for this overlap check; test
images and labels are not opened by the training script.

## Installation

Python 3.8 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The core dependencies are PyTorch, NumPy, SciPy, NiBabel, and
`ml-collections`. GPU execution requires a CUDA-compatible PyTorch build.

## Training

The reported LPBA40 configuration uses 500 epochs, batch size 1, Adam with an
initial learning rate of `1e-4`, polynomial learning-rate decay, voxelwise MSE,
and a first-order displacement smoothness weight of `0.02`. Validation DSC
alone controls selection of `best.pt`.

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

Use `--model paramorph-c` for ParaMorph-C and `--model paramorph-small` for
ParaMorph-S. The manuscript reports ParaMorph-S with seed 47 only. TransMorph
and full ParaMorph were trained independently with seeds 47, 726, and 2026 for
the repeated-training analysis.

### Optional augmentation

Data augmentation is disabled by default and was **not used** for the reported
LPBA40 results. The loader retains synchronized random flips and small-angle
rotations as optional experiments:

```bash
--augment --rotation-degrees 5 --rotation-probability 0.5
```

The sampled transform is shared by the moving and fixed images. Validation and
test data are never augmented. Scaling, elastic deformation, gamma
augmentation, and intensity perturbation are not implemented or claimed for
the reported experiments.

## Held-out testing

Run the validation-selected checkpoint after model selection is complete:

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

The script writes case-level results to `per_case_metrics.csv` and aggregate
means and sample standard deviations to `summary.json`.

## Metrics

- **DSC:** calculated for each evaluated foreground label and macro-averaged
  within each registration pair.
- **ASSD:** the mean of the two directed surface distances. LPBA40 spacing is
  `(1, 1, 1)`, so values are reported in millimetres.
- **Folding percentage:** `100 x mean(det(J_phi) <= 0)`, where `J_phi` is the
  spatial Jacobian of the identity-plus-displacement mapping.

LPBA40 evaluation uses the 54 foreground IDs defined in `metrics.py`. Raw IDs
181 and 182 are excluded, matching the manuscript evaluation protocol.

## Repository contents

```text
Paramorph/
  data/
  models/
  splits/
  metrics.py
  train_paramorph.py
  test_paramorph.py
  requirements.txt
  README.md
```

Generated caches, experiment logs, checkpoints, and datasets should not be
committed to the repository.
