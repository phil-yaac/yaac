"""Tests for the YAD model."""

import pytest
import torch

from yaac.models.yad.yad import YAD, make_yad, _make_anchor_points


def test_make_yad_shapetype():
    """Test that make_yad creates a model with correct type and structure."""
    model = make_yad(num_classes=80)
    assert isinstance(model, YAD)
    assert isinstance(model, torch.nn.Module)


def test_yad_forward_shapetype():
    """Test that the forward pass outputs have correct shapes and types."""
    num_classes = 80
    model = make_yad(num_classes=num_classes)
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)
    bbox_out, cls_out, objectness_out = model(image)

    # Check types
    assert isinstance(bbox_out, torch.Tensor)
    assert isinstance(cls_out, torch.Tensor)
    assert isinstance(objectness_out, torch.Tensor)

    # Check shapes
    assert bbox_out.shape[0] == batch_size
    assert cls_out.shape[0] == batch_size
    assert objectness_out.shape[0] == batch_size
    assert cls_out.shape[1] == num_classes
    assert bbox_out.shape[1] == 4
    assert objectness_out.shape[1] == 1


def test_make_anchor_points_values():
    """Test that anchor points are placed correctly in normalized coordinates."""
    # Input parameters - using a small 3x3 grid for easy verification
    image_height = 100
    image_width = 100
    feature_map_height = 3
    feature_map_width = 3

    # Expected outputs - hard-coded for a 3x3 grid
    expected_points = torch.tensor([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],  # First row
        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],  # Middle row
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],  # Last row
    ])

    # Call function under test
    anchor_points = _make_anchor_points(
        image_height=image_height,
        image_width=image_width,
        feature_map_height=feature_map_height,
        feature_map_width=feature_map_width,
    )

    # Verify outputs
    assert anchor_points.shape == (9, 2)
    assert torch.allclose(anchor_points, expected_points) 