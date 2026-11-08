"""ConvNet-only component-isolation model used in Tables 6 and 7."""

import torch

from .paramorph import ParaMorph


class ConvNetOnly(ParaMorph):
    """Remove the Transformer pathway while retaining the common decoder.

    The ConvNet hierarchy supplies the six decoder-resolution tensors directly:
    ``cov5`` is the decoder bottleneck and ``cov4`` through ``cov0`` are the
    successive skip inputs.  No local/global fusion or GF refinement is used.
    """

    def __init__(self, config):
        super().__init__(config)
        # Remove the unused Transformer parameters from the ablation model and
        # its checkpoints rather than merely bypassing them in ``forward``.
        self.transformer = None

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, D, H, W] input, got {tuple(x.shape)}")

        source = x[:, 0:1, ...]
        cov0_native = self.conv1(x)
        cov1 = self.down1(cov0_native)
        cov0 = self.Conv_1x1(cov0_native)
        cov2 = self.down2(self.conv2(cov1))
        cov3 = self.down3(self.conv3(cov2))
        cov4 = self.down4(self.conv4(cov3))
        cov5 = self.down5(self.conv5(cov4))

        decoded = self.up0(cov5, cov4)
        decoded = self.up1(decoded, cov3)
        decoded = self.up2(decoded, cov2)
        decoded = self.up3(decoded, cov1)
        decoded = self.up4(decoded, cov0)
        flow = self.reg_head(decoded)
        warped = self.spatial_trans(source, flow)
        return warped, flow
