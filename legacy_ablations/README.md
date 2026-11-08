# Legacy component-isolation implementations

This directory preserves the model sources and training entry points used for
the component-isolation experiments reported in manuscript Tables 6 and 7.
They are retained separately from the cleaned ParaMorph release so that the
historical ablation paths are not silently refactored.

The authoritative row-to-source mapping is
`../configs/ablation_manifest.json`. In particular:

- `Combined encoders w/o GF/PMFE` uses
  `models/swin_conv_fusion_ab_cov1.py`. Its active forward path combines the
  four resolution-matched Transformer and ConvNet feature maps by element-wise
  addition. The concatenate-and-project block present in the file is inside a
  disabled triple-quoted experimental block and was not used.
- `Baseline + GF` uses `models/TransMorph_gap.py`. The three Transformer skip
  features (`f1`, `f2`, and `f3`) are passed through separate residual 3D
  convolutional refinement blocks. The deepest Transformer feature and the two
  original shallow TransMorph convolutional skips are retained.
- `Baseline + PMFE` uses `models/swin_conv_fusion.py`.
- Full ParaMorph and ParaMorph-C use
  `models/swin_conv_fusion_cov3_gap.py` and
  `models/swin_conv_mixed_gap.py`, respectively.

The ConvNet-encoder-only row is implemented in the cleaned release as
`../models/convnet_only.py` and is available from both primary entry points
with `--model convnet-only`.

The legacy training files retain the original dataset-path variables. Set
`train_dir`, `val_dir`, and `save_dir` in the corresponding entry point before
running. Dataset membership must follow the released split files in
`../splits/`. All LPBA40 learning-based variants use the same image-similarity
weight of 1 and deformation-smoothness weight of 0.02.

These files are supplied for architectural traceability. The cleaned primary
entry points at the repository root remain the recommended interface for
training and testing the released ParaMorph, ParaMorph-C, and ParaMorph-S
models.
