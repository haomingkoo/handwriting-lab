import torch

from mnist import data_prep


def test_datasets_mnistdataset():
    """Test for MNISTDataset class."""

    dataset = data_prep.datasets.MNISTDataset(
        data_dir_path="./src/tests/sample_mnist_data_for_tests",
        anno_file_name="sample_data_table.csv",
        to_grayscale=False,
        to_tensor=False,
        transform=data_prep.transforms.MNIST_TRANSFORM_STEPS["train"],
    )

    assert len(dataset) == 3 and dataset[0][0] == "train/0/16585.png"


def test_build_train_augmentation_disabled():
    """Augmentation should be optional."""
    transform = data_prep.transforms.build_train_augmentation(enabled=False)
    assert transform is None


def test_build_train_augmentation_rotation_shape():
    """Rotation augment should preserve tensor shape."""
    transform = data_prep.transforms.build_train_augmentation(
        enabled=True,
        rotation_degrees=45,
        rotation_prob=1.0,
        invert_prob=0.0,
    )

    image = torch.zeros((1, 28, 28))
    output = transform(image)
    assert output.shape == image.shape


def test_build_train_augmentation_invert():
    """Invert augment should flip pixel values when probability is 1."""
    transform = data_prep.transforms.build_train_augmentation(
        enabled=True,
        rotation_degrees=0,
        rotation_prob=0.0,
        invert_prob=1.0,
    )

    image = torch.zeros((1, 28, 28))
    output = transform(image)
    assert torch.equal(output, torch.ones((1, 28, 28)))


def test_build_train_augmentation_affine_perspective_shape():
    """Affine and perspective augments should keep the expected tensor shape."""
    transform = data_prep.transforms.build_train_augmentation(
        enabled=True,
        rotation_degrees=0,
        rotation_prob=0.0,
        affine_prob=1.0,
        affine_translate_x=0.1,
        affine_translate_y=0.1,
        affine_scale_min=0.9,
        affine_scale_max=1.1,
        affine_shear_degrees=8,
        perspective_prob=1.0,
        perspective_distortion_scale=0.2,
        invert_prob=0.0,
    )

    image = torch.zeros((1, 28, 28))
    output = transform(image)
    assert output.shape == image.shape
