"""Definition of transforms sequence for data preparation."""

from __future__ import annotations

import torchvision


def _clamp_probability(probability: float) -> float:
    """Bound probabilities so config mistakes do not crash transform creation."""
    return max(0.0, min(1.0, float(probability)))


def build_train_augmentation(
    *,
    enabled: bool = False,
    rotation_degrees: float = 0.0,
    rotation_prob: float = 0.0,
    affine_prob: float = 0.0,
    affine_translate_x: float = 0.0,
    affine_translate_y: float = 0.0,
    affine_scale_min: float = 1.0,
    affine_scale_max: float = 1.0,
    affine_shear_degrees: float = 0.0,
    perspective_prob: float = 0.0,
    perspective_distortion_scale: float = 0.0,
    invert_prob: float = 0.0,
) -> torchvision.transforms.Compose | None:
    """Build optional training augmentations for rotation/inversion robustness."""
    if not enabled:
        return None

    steps: list[object] = []

    if float(rotation_degrees) > 0 and float(rotation_prob) > 0:
        steps.append(
            torchvision.transforms.RandomApply(
                [
                    torchvision.transforms.RandomRotation(
                        degrees=float(rotation_degrees), fill=0
                    )
                ],
                p=_clamp_probability(rotation_prob),
            )
        )

    if float(affine_prob) > 0:
        translate = (
            max(0.0, min(1.0, float(affine_translate_x))),
            max(0.0, min(1.0, float(affine_translate_y))),
        )
        scale_range = (
            min(float(affine_scale_min), float(affine_scale_max)),
            max(float(affine_scale_min), float(affine_scale_max)),
        )
        steps.append(
            torchvision.transforms.RandomApply(
                [
                    torchvision.transforms.RandomAffine(
                        degrees=0.0,
                        translate=translate,
                        scale=scale_range,
                        shear=float(affine_shear_degrees),
                        fill=0,
                    )
                ],
                p=_clamp_probability(affine_prob),
            )
        )

    if float(perspective_prob) > 0 and float(perspective_distortion_scale) > 0:
        steps.append(
            torchvision.transforms.RandomPerspective(
                distortion_scale=max(
                    0.0, min(1.0, float(perspective_distortion_scale))
                ),
                p=_clamp_probability(perspective_prob),
                fill=0,
            )
        )

    if float(invert_prob) > 0:
        steps.append(
            torchvision.transforms.RandomInvert(p=_clamp_probability(invert_prob))
        )

    if not steps:
        return None

    return torchvision.transforms.Compose(steps)


MNIST_TRANSFORM_STEPS = {
    "train": torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(32),
            torchvision.transforms.RandomCrop((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    "test": torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(32),
            torchvision.transforms.CenterCrop((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
}
