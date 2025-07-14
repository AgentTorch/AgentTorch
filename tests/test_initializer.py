import pytest
import torch
from agent_torch.core.initializer import Initializer


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {"simulation_metadata": {"device": "cpu"}}


@pytest.fixture
def mock_registry():
    """Mock registry for testing"""

    class MockRegistry:
        pass

    return MockRegistry()


@pytest.fixture
def initializer(mock_config, mock_registry):
    """Create an initializer instance for testing"""
    return Initializer(mock_config, mock_registry)


def test_initialize_from_default_list_with_matching_shape(initializer):
    """
    Test that list values with matching shapes work correctly.
    """
    result = initializer._initialize_from_default([0.0, 0.0], [1000, 2])

    assert result.shape == torch.Size([1000, 2])
    assert torch.allclose(result[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(result[-1], torch.tensor([0.0, 0.0]))
    assert torch.all(result == result[0])


def test_initialize_from_default_list_3d_case(initializer):
    """
    Test the specific case mentioned by user: [0, 0, 0] with shape [1000, 3].
    """
    result = initializer._initialize_from_default([0, 0, 0], [1000, 3])

    assert result.shape == torch.Size([1000, 3])
    assert torch.allclose(result[0], torch.tensor([0, 0, 0]))
    assert torch.allclose(result[-1], torch.tensor([0, 0, 0]))
    assert torch.all(result == result[0])


def test_initialize_from_default_list_non_zero_values(initializer):
    """
    Test with non-zero values to ensure proper broadcasting.
    """
    result = initializer._initialize_from_default([1.0, 2.0, 3.0], [500, 3])

    assert result.shape == torch.Size([500, 3])
    assert torch.allclose(result[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(result[-1], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.all(result == result[0])


def test_initialize_from_default_single_value(initializer):
    """
    Test that single values still work with the existing behavior.
    """
    result = initializer._initialize_from_default(5.0, [100, 2])

    assert result.shape == torch.Size([100, 2])
    assert torch.all(result == 5.0)


def test_initialize_from_default_dimension_mismatch(initializer):
    """
    Test graceful fallback when list dimensions don't match target shape.
    """
    result = initializer._initialize_from_default([1, 2, 3, 4], [100, 2])

    assert result.shape == torch.Size([100, 2])
    assert torch.all(result == 1.0)  # Should use first value


def test_initialize_from_default_string_passthrough(initializer):
    """
    Test that string values are passed through unchanged.
    """
    result = initializer._initialize_from_default("test_string", [10, 2])

    assert result == "test_string"


def test_initialize_from_default_list_no_shape(initializer):
    """
    Test that lists without explicit shape requirements work.
    """
    result = initializer._initialize_from_default([1.0, 2.0], None)

    assert torch.allclose(result, torch.tensor([1.0, 2.0]))


def test_initialize_from_default_complex_broadcasting(initializer):
    """
    Test more complex broadcasting scenarios.
    """
    # Test 1D list with 3D target shape
    result = initializer._initialize_from_default([1.0, 2.0], [10, 5, 2])

    assert result.shape == torch.Size([10, 5, 2])
    assert torch.allclose(result[0, 0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(result[-1, -1], torch.tensor([1.0, 2.0]))


def test_initialize_from_default_different_dtypes(initializer):
    """
    Test that different data types are handled correctly.
    """
    # Integer values
    result_int = initializer._initialize_from_default([1, 2], [100, 2])
    assert result_int.shape == torch.Size([100, 2])
    assert torch.allclose(result_int[0], torch.tensor([1, 2]))

    # Float values
    result_float = initializer._initialize_from_default([1.5, 2.5], [100, 2])
    assert result_float.shape == torch.Size([100, 2])
    assert torch.allclose(result_float[0], torch.tensor([1.5, 2.5]))


def test_initialize_from_default_edge_cases(initializer):
    """
    Test edge cases and boundary conditions.
    """
    # Empty list
    result_empty = initializer._initialize_from_default([], [10, 0])
    assert result_empty.shape == torch.Size([10, 0])

    # Single element list with larger shape
    result_single = initializer._initialize_from_default([42.0], [5, 3])
    assert result_single.shape == torch.Size([5, 3])
    assert torch.all(result_single == 42.0)


def test_device_consistency(initializer):
    """
    Test that tensors are moved to the correct device.
    """
    result = initializer._initialize_from_default([1.0, 2.0], [10, 2])

    # Should be on CPU device as specified in mock_config
    assert result.device == torch.device("cpu")
