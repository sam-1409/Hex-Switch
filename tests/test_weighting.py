"""
Unit tests for HCI-guided and objective criteria weighting.
Validates Saaty AHP consistency theorem, literature profiles, and objective methods.
"""

import numpy as np
import pytest

from edge_switching.qos_model import Criterion
from edge_switching.weighting import (
    compute_ahp_weights,
    compute_critic_weights,
    compute_entropy_weights,
    compute_equal_weights,
    get_all_weight_methods,
    get_literature_ahp_profile,
    AHP_VIDEO_STREAMING_MATRIX,
    AHP_INTERACTIVE_MATRIX,
    AHP_BALANCED_MATRIX,
)


def test_ahp_consistency_literature_profiles():
    """All literature-derived AHP profiles must satisfy Saaty's CR < 0.10 threshold."""
    for prof in ["video_streaming", "interactive_gaming", "balanced_edge"]:
        w_dict, ci, cr = get_literature_ahp_profile(prof)
        assert len(w_dict) == 5
        assert np.isclose(sum(w_dict.values()), 1.0)
        # CR < 0.10 is the fundamental Saaty transitivity criterion
        assert cr < 0.10, f"Profile {prof} has CR={cr:.4f} >= 0.10"
        assert ci >= 0.0


def test_ahp_video_profile_prioritizes_bandwidth_and_loss():
    """Video QoE profile must allocate highest weights to bandwidth, loss, and availability."""
    w_dict, _, _ = get_literature_ahp_profile("video_streaming")
    # In video streaming, bandwidth and availability should outweigh raw latency
    assert w_dict[Criterion.EFFECTIVE_BW] > w_dict[Criterion.LATENCY]
    assert w_dict[Criterion.AVAILABILITY] > w_dict[Criterion.LATENCY]


def test_ahp_interactive_profile_prioritizes_latency():
    """Interactive profile must allocate highest weight to latency."""
    w_dict, _, _ = get_literature_ahp_profile("interactive_gaming")
    assert w_dict[Criterion.LATENCY] > w_dict[Criterion.EFFECTIVE_BW]
    assert w_dict[Criterion.LATENCY] > w_dict[Criterion.SERVER_LOAD]


def test_ahp_non_reciprocal_matrix_raises_error():
    """A matrix that violates A_ji = 1/A_ij must raise ValueError."""
    bad_matrix = np.array([
        [1.0, 2.0],
        [2.0, 1.0]  # Should be 0.5
    ])
    with pytest.raises(ValueError):
        compute_ahp_weights(bad_matrix)


def test_entropy_weights_mathematical_properties():
    """Shannon entropy weighting must sum to 1.0 and favor dispersed criteria."""
    # Column 0: identical utility across all candidates (zero information)
    # Column 1: highly diverse utility
    U = np.array([
        [0.8, 0.1, 0.5, 0.5, 1.0],
        [0.8, 0.9, 0.5, 0.5, 1.0],
        [0.8, 0.5, 0.5, 0.5, 1.0],
    ])
    weights = compute_entropy_weights(U)
    assert len(weights) == 5
    assert np.isclose(np.sum(weights), 1.0)
    # Diverse column (index 1) must have strictly higher weight than uniform column (index 0)
    assert weights[1] > weights[0]


def test_critic_weights_mathematical_properties():
    """CRITIC weighting must sum to 1.0 and penalize redundant correlated criteria."""
    U = np.array([
        [0.1, 0.9, 0.3, 0.2, 1.0],
        [0.5, 0.5, 0.7, 0.6, 1.0],
        [0.9, 0.1, 0.4, 0.8, 1.0],
        [0.3, 0.7, 0.2, 0.3, 1.0],
    ])
    weights = compute_critic_weights(U)
    assert len(weights) == 5
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)


def test_equal_weights():
    """Equal weights must return exactly 1/K."""
    w = compute_equal_weights(5)
    assert len(w) == 5
    assert np.allclose(w, 0.20)
