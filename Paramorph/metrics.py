"""Registration metrics with an explicit moving-to-fixed convention."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


LPBA40_LABELS = (
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62, 63, 64,
    65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    91, 92, 101, 102, 121, 122, 161, 162, 163, 164, 165, 166,
)


def dice_per_label(prediction, target, labels=LPBA40_LABELS):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    values = []
    for label in labels:
        pred = prediction == label
        true = target == label
        denom = pred.sum() + true.sum()
        values.append(np.nan if denom == 0 else 2.0 * np.logical_and(pred, true).sum() / denom)
    return np.asarray(values, dtype=np.float64)


def _surface(mask):
    return np.logical_xor(mask, binary_erosion(mask, border_value=0))


def assd_per_label(prediction, target, spacing=(1.0, 1.0, 1.0), labels=LPBA40_LABELS):
    """Symmetric mean surface distance in physical units defined by spacing."""
    values = []
    for label in labels:
        pred = np.asarray(prediction) == label
        true = np.asarray(target) == label
        if not pred.any() or not true.any():
            values.append(np.nan)
            continue
        pred_surface, true_surface = _surface(pred), _surface(true)
        to_true = distance_transform_edt(~true_surface, sampling=spacing)[pred_surface]
        to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[true_surface]
        values.append(float(0.5 * (to_true.mean() + to_pred.mean())))
    return np.asarray(values, dtype=np.float64)


def jacobian_determinant(displacement):
    """Determinant of I + grad(u) for flow shaped [3, D, H, W]."""
    displacement = np.asarray(displacement, dtype=np.float64)
    if displacement.ndim != 4 or displacement.shape[0] != 3:
        raise ValueError(f"Expected displacement [3, D, H, W], got {displacement.shape}")
    coordinates = np.stack(
        np.meshgrid(*[np.arange(n, dtype=np.float64) for n in displacement.shape[1:]], indexing="ij"),
        axis=0,
    )
    mapping = coordinates + displacement
    jacobian = np.empty((*displacement.shape[1:], 3, 3), dtype=np.float64)
    for component in range(3):
        derivatives = np.gradient(mapping[component], edge_order=1)
        for axis, derivative in enumerate(derivatives):
            jacobian[..., component, axis] = derivative
    return np.linalg.det(jacobian)


def nonpositive_jacobian_percentage(displacement):
    """Percentage of voxels with a non-positive deformation Jacobian."""
    return float(100.0 * np.mean(jacobian_determinant(displacement) <= 0.0))


def warp_labels(labels, flow):
    """Warp moving labels toward fixed space with nearest interpolation."""
    from models.paramorph import SpatialTransformer

    transformer = SpatialTransformer(flow.shape[2:], mode="nearest").to(flow.device)
    return transformer(labels.float(), flow).long()
