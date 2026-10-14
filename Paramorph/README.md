# ParaMorph — LPBA40 reproducible training and evaluation

This folder contains ParaMorph and its ParaMorph-C cross-interaction variant,
together with their LPBA40 training, validation, and held-out test pipeline.
The public release is intentionally scoped to LPBA40 as a self-contained
reference implementation. It does not include trained checkpoints or complete
dataset-specific execution packages for HPH-PM and Prostate158; the atlas
identities, participant splits, preprocessing, and evaluation protocols for
those datasets are documented in the manuscript.
The code uses one explicit convention:
each participant is the **moving** image and `S01` is the **fixed** atlas; the
predicted displacement warps moving coordinates into fixed-atlas space. This is
subject-to-atlas registration. Use the same wording in the manuscript and do
not call this direction atlas-to-subject.

Both released networks contain 63,328,195 trainable parameters. ParaMorph-C
adds the channel-aligned ConvNet tokens to the corresponding Transformer-stage
outputs before normalization, without adding trainable layers. It retains the
same scale-wise decoder aggregation as ParaMorph. The count excludes obsolete
development layers that were never called in `forward`.

For the reviewer-requested capacity-matched control, `paramorph-small`
(ParaMorph-S) retains all ParaMorph pathways, GF aggregation, PMFE refinement,
and decoder stages, but reduces the shared base width from 96 to 84 channels.
It contains 48,569,521 trainable parameters, 3.90% more than the manuscript's
46.746M-parameter TransMorph comparator and therefore falls within a prespecified
5% parameter-matching tolerance.

## Data and split contract

The supplied LPBA40 images and labels are `160 x 192 x 160` voxels
at `1 mm` isotropic spacing. The scripts intentionally refuse mismatched shapes
instead of silently resizing images or labels. The expected layout is:

```text
LPBA40/
  train/S11.delineation.skullstripped.nii.gz ... S40...
  test/S02.delineation.skullstripped.nii.gz ... S10...
  label/S01.delineation.structure.label.nii.gz
  label/S11.delineation.structure.label.nii.gz ... S40...
  label/S02.delineation.skullstripped.nii.gz ... S10...
  fixed.nii.gz
```

For compatibility with commonly distributed LPBA40 layouts, label volumes are
also accepted when all cases use the
`SXX.delineation.structure.label.nii.gz` naming convention. The loader first
checks that standard label name and then checks the `skullstripped` filename in
the label directory. This filename fallback does not alter image data, label
values, preprocessing, or evaluation.

The released split preserves the original LPBA40 test cohort used by this code
base: S01 is the atlas, S02--S10 are held-out test participants, and the former
S11--S40 training cohort is divided into S11--S35 for training and S36--S40 for
validation. Split files are
versioned under `splits/`; the training script checks all three for overlap.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Only validation DSC controls `best.pt`. The test split is read as identifiers
solely for the disjointness check; test images and labels are never opened.
Data augmentation is disabled by default, matching the reported LPBA40
experiments.

```bash
python train_paramorph.py \
  --images-dir /path/LPBA40/train \
  --labels-dir /path/LPBA40/label \
  --atlas-image /path/LPBA40/fixed.nii.gz \
  --atlas-label /path/LPBA40/label/S01.delineation.structure.label.nii.gz \
  --train-split splits/lpba40_train.txt \
  --val-split splits/lpba40_validation.txt \
  --test-split splits/lpba40_test.txt \
  --output-dir runs/lpba40_seed47 --seed 47 --model paramorph
```

To train ParaMorph-C, use the same command with `--model paramorph-c` and a
separate output directory.

To run the capacity-matched control, use the identical command, split, seed,
preprocessing, optimizer, and stopping rule with `--model paramorph-small`.
The ParaMorph-S control reported in the manuscript uses seed `47` only. Train
TransMorph and full ParaMorph independently with seeds `47`, `726`, and `2026`
in separate output directories for the repeated-training analysis. Do not tune
any model on the held-out test set. Checkpoints have stable names (`best.pt` and
`last.pt`) and record the split identifiers, seed, image shape, flow convention,
and selection metric.

### Optional augmentation

The loader also supports synchronized random flips and small-angle rotations as
an optional experiment. This option was not used for the results reported in
the manuscript. Append these flags to the training command:

```bash
--augment --rotation-degrees 5 --rotation-probability 0.5
```

When enabled, each axis is flipped independently with probability 0.5, and one
random anatomical plane is rotated by an angle sampled from -5 to 5 degrees
with probability 0.5. The same transform is applied to the moving image and
fixed atlas. Validation and test data are never augmented. The applied settings
are stored in each checkpoint.

## Held-out testing

Run the locked validation-selected checkpoint once after training decisions are
complete:

```bash
python test_paramorph.py \
  --images-dir /path/LPBA40/test \
  --labels-dir /path/LPBA40/label \
  --atlas-image /path/LPBA40/fixed.nii.gz \
  --atlas-label /path/LPBA40/label/S01.delineation.structure.label.nii.gz \
  --test-split splits/lpba40_test.txt \
  --checkpoint runs/lpba40_seed47/best.pt \
  --output-dir results/lpba40_seed47 --model paramorph
```

For a ParaMorph-C checkpoint, pass `--model paramorph-c`.

The test script writes per-case DSC, ASSD in millimetres, and the percentage of
non-positive Jacobian determinants to `per_case_metrics.csv`, plus aggregate
mean and sample standard deviation to `summary.json`. LPBA40 evaluation uses the
54 label IDs listed in `metrics.py`; the manuscript must report 54 evaluated
labels unless a different, documented label map is actually used.
The 54-ID convention follows the public ModeT LPBA40 protocol and evaluation
code; raw IDs 181 and 182 are not part of that published 54-ROI mapping.

## Geometry tests

```bash
pytest -q
```

The Jacobian determinant is evaluated for the mapping formed by adding the
displacement field to the identity grid, as in the public ModeT implementation.
Following the percentage-based reporting convention in the ModeT paper, the
voxel ratio `mean(det(J_phi) <= 0)` is multiplied by 100. Thus, a reported value
of 0.004 denotes 0.004% of evaluated voxels with non-positive determinants.
The tests verify that identity and
translation fields have unit Jacobian and that a known diagonal scaling has the
analytic determinant. Synthetic warping
tests additionally verify that images and labels follow the same moving-to-fixed
sampling direction, that identity warping is exact, and that nearest-neighbor
label interpolation introduces no new class IDs. Together they address the
derivative-index and transformation-direction issues raised during peer review.

The repository was smoke-tested with one gradient update, one validation case,
checkpoint saving, and a separate held-out inference pass on 64-cubed crops
from the supplied LPBA40 NIfTI files. Full experiments use the original
`160 x 192 x 160` volumes.
