"""Configuration for the released ParaMorph architecture."""

from ml_collections import ConfigDict


def get_paramorph_config(image_shape=(160, 192, 160)):
    """Return the architecture used for ParaMorph.

    ``image_shape`` follows array order (D, H, W). It must match the
    preprocessed NIfTI volumes supplied to the training and test scripts.
    """
    config = ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 5)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0.0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = tuple(image_shape)
    return config


def get_paramorph_small_config(image_shape=(160, 192, 160)):
    """Capacity-matched ParaMorph-S configuration for TransMorph comparison.

    ParaMorph-S retains every proposed pathway and fusion/refinement component
    while reducing the shared base width from 96 to 84. It has 48,569,521
    trainable parameters, within 4% of the 46,746,000-parameter TransMorph
    comparator reported in the manuscript.
    """
    config = get_paramorph_config(image_shape)
    config.embed_dim = 84
    return config
