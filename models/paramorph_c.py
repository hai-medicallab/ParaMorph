"""ParaMorph-C: ParaMorph with parameter-free cross-pathway interaction."""

from .paramorph import ParaMorph


class ParaMorphC(ParaMorph):
    """Add aligned ConvNet tokens to each Transformer stage before normalization.

    ParaMorph's original scale-wise decoder aggregation is retained.  The cross
    interaction introduces no trainable layers, so ParaMorph and ParaMorph-C
    have identical parameter counts and state-dict layouts.
    """

    def forward(self, x):
        return self._forward_impl(x, cross_pathway=True)
