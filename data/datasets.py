"""LPBA40 atlas-registration datasets with explicit, disjoint split files."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import rotate
from torch.utils.data import Dataset


def read_split(path):
    """Read one case name per line, ignoring blanks and comments."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    cases = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not cases:
        raise ValueError(f"Split file is empty: {path}")
    if len(cases) != len(set(cases)):
        raise ValueError(f"Split file contains duplicate cases: {path}")
    return cases


def assert_disjoint_splits(**splits):
    """Fail fast if any participant occurs in more than one split."""
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap:
                raise ValueError(f"{left}/{right} overlap: {sorted(overlap)}")


def _resolve_case(root, case, filename_template):
    path = Path(case)
    if not path.is_absolute():
        filename = case if str(case).endswith(".nii.gz") else filename_template.format(case=case)
        path = Path(root) / filename
    if path.exists():
        return path
    raise FileNotFoundError(path)


def _resolve_label_case(root, case, filename_template):
    """Resolve an LPBA40 label while supporting both distributed filenames.

    Most cases use ``*.delineation.structure.label.nii.gz``.  Some prepared
    LPBA40 copies retain ``*.delineation.skullstripped.nii.gz`` for the same
    discrete label volumes in the label directory.  The primary template is
    preferred, and the alternate name is used only when it is absent.
    """
    try:
        return _resolve_case(root, case, filename_template)
    except FileNotFoundError as primary_error:
        case_path = Path(case)
        if case_path.is_absolute() or str(case).endswith(".nii.gz"):
            raise primary_error
        alternate = Path(root) / f"{case}.delineation.skullstripped.nii.gz"
        if alternate.exists():
            return alternate
        raise primary_error


def _load_volume(path, dtype):
    volume = np.asarray(nib.load(str(path)).dataobj, dtype=dtype)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume at {path}, got shape {volume.shape}")
    return volume


def _minmax(volume):
    lo, hi = float(volume.min()), float(volume.max())
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - lo) / (hi - lo)).astype(np.float32, copy=False)


def synchronized_random_flip(volumes, flip_axes=None):
    """Apply one shared random flip decision per spatial axis to paired images.

    Each axis is independently flipped with probability 0.5, and the sampled
    transform is shared by the fixed and moving images.
    """
    if flip_axes is None:
        flip_axes = tuple(bool(value) for value in (torch.rand(3) < 0.5).tolist())
    if len(flip_axes) != 3:
        raise ValueError("flip_axes must contain three booleans")
    augmented = []
    for volume in volumes:
        result = volume
        for axis, should_flip in enumerate(flip_axes):
            if should_flip:
                result = np.flip(result, axis=axis)
        augmented.append(np.ascontiguousarray(result))
    return augmented


def synchronized_random_rotation(volumes, max_degrees, probability=0.5,
                                 sampled=None):
    """Apply one small 3-D rotation to every image in a registration pair."""
    if max_degrees < 0:
        raise ValueError("max_degrees must be non-negative")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("rotation probability must be in [0, 1]")
    if sampled is None:
        apply_rotation = bool(torch.rand(()) < probability)
        plane = ((0, 1), (0, 2), (1, 2))[int(torch.randint(0, 3, ()).item())]
        angle = float((torch.rand(()) * 2.0 - 1.0) * max_degrees)
    else:
        apply_rotation, plane, angle = sampled
    if not apply_rotation or max_degrees == 0:
        return [np.ascontiguousarray(volume) for volume in volumes]
    return [
        np.ascontiguousarray(rotate(
            volume, angle=float(angle), axes=tuple(plane), reshape=False,
            order=1, mode="constant", cval=0.0, prefilter=False,
        ).astype(np.float32, copy=False))
        for volume in volumes
    ]


class LPBA40AtlasDataset(Dataset):
    """Register each split participant (moving) to one fixed atlas.

    Labels are loaded only for validation/test. Training never reads labels.
    All volumes must already have the documented preprocessing and shape.
    """

    def __init__(self, cases, images_dir, atlas_image, image_shape,
                 labels_dir=None, atlas_label=None, normalize=True,
                 random_flip=False,
                 rotation_degrees=0.0, rotation_probability=0.0,
                 image_template="{case}.delineation.skullstripped.nii.gz",
                 label_template="{case}.delineation.structure.label.nii.gz"):
        self.cases = list(cases)
        self.images_dir = Path(images_dir)
        self.atlas_image_path = Path(atlas_image)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.atlas_label_path = Path(atlas_label) if atlas_label else None
        self.image_shape = tuple(image_shape)
        self.normalize = normalize
        self.random_flip = random_flip
        self.rotation_degrees = float(rotation_degrees)
        self.rotation_probability = float(rotation_probability)
        self.image_template = image_template
        self.label_template = label_template

        if (self.labels_dir is None) != (self.atlas_label_path is None):
            raise ValueError("labels_dir and atlas_label must be supplied together")
        augmentation_enabled = self.random_flip or self.rotation_probability > 0
        if self.labels_dir is not None and augmentation_enabled:
            raise ValueError("Training augmentation must not be enabled for validation/test datasets")

        fixed = _load_volume(self.atlas_image_path, np.float32)
        self._check_shape(fixed, self.atlas_image_path)
        self.fixed = _minmax(fixed) if normalize else fixed.astype(np.float32, copy=False)

        if self.atlas_label_path:
            self.fixed_label = _load_volume(self.atlas_label_path, np.int16)
            self._check_shape(self.fixed_label, self.atlas_label_path)
        else:
            self.fixed_label = None

    def _check_shape(self, volume, path):
        if tuple(volume.shape) != self.image_shape:
            raise ValueError(
                f"Shape mismatch for {path}: expected {self.image_shape}, got {tuple(volume.shape)}. "
                "Preprocess the data before running this code; implicit resizing is disabled."
            )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        moving_path = _resolve_case(self.images_dir, case, self.image_template)
        moving = _load_volume(moving_path, np.float32)
        self._check_shape(moving, moving_path)
        if self.normalize:
            moving = _minmax(moving)

        fixed = self.fixed
        if self.random_flip:
            moving, fixed = synchronized_random_flip((moving, fixed))
        if self.rotation_probability > 0:
            moving, fixed = synchronized_random_rotation(
                (moving, fixed), self.rotation_degrees, self.rotation_probability
            )
        sample = {
            "case_id": moving_path.name.removesuffix(".nii.gz"),
            "moving": torch.from_numpy(np.ascontiguousarray(moving[None])).float(),
            "fixed": torch.from_numpy(np.ascontiguousarray(fixed[None])).float(),
        }

        if self.fixed_label is not None:
            moving_label_path = _resolve_label_case(
                self.labels_dir, case, self.label_template
            )
            moving_label = _load_volume(moving_label_path, np.int16)
            self._check_shape(moving_label, moving_label_path)
            sample["moving_label"] = torch.from_numpy(
                np.ascontiguousarray(moving_label[None].astype(np.int64))
            )
            sample["fixed_label"] = torch.from_numpy(
                np.ascontiguousarray(self.fixed_label[None].astype(np.int64))
            )
        return sample
