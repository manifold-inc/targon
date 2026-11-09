"""Tests for WeightRequest validation and API endpoints."""

import math
import pytest
from pydantic import ValidationError

# We test the models and validation logic directly without needing
# bittensor installed, by importing only what we need.
import sys
from unittest.mock import MagicMock

# Mock heavy dependencies so tests can run without bittensor installed
sys.modules.setdefault("bittensor", MagicMock())
sys.modules.setdefault("bittensor.core", MagicMock())
sys.modules.setdefault("bittensor.core.axon", MagicMock())
sys.modules.setdefault("bittensor_wallet", MagicMock())
sys.modules.setdefault("bt", MagicMock())

from main import WeightRequest, validate_weight_value, MAX_UIDS


class TestValidateWeightValue:
    """Tests for the validate_weight_value helper function."""

    def test_valid_integer(self):
        assert validate_weight_value(5) == 5

    def test_valid_float(self):
        assert validate_weight_value(0.5) == 0.5

    def test_valid_zero(self):
        assert validate_weight_value(0) == 0

    def test_valid_zero_float(self):
        assert validate_weight_value(0.0) == 0.0

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            validate_weight_value(float("nan"))

    def test_rejects_positive_infinity(self):
        with pytest.raises(ValueError, match="finite"):
            validate_weight_value(float("inf"))

    def test_rejects_negative_infinity(self):
        with pytest.raises(ValueError, match="finite"):
            validate_weight_value(float("-inf"))

    def test_rejects_negative_integer(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_weight_value(-1)

    def test_rejects_negative_float(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_weight_value(-0.5)


class TestWeightRequestValidation:
    """Tests for WeightRequest pydantic model validation."""

    def test_valid_request(self):
        req = WeightRequest(uids=[0, 1, 2], weights=[0.5, 0.3, 0.2], version=1)
        assert req.uids == [0, 1, 2]
        assert req.weights == [0.5, 0.3, 0.2]
        assert req.version == 1

    def test_valid_request_integer_weights(self):
        req = WeightRequest(uids=[0, 1], weights=[1, 2], version=0)
        assert req.weights == [1, 2]

    def test_valid_request_mixed_weights(self):
        req = WeightRequest(uids=[0, 1], weights=[1, 0.5], version=1)
        assert req.weights == [1, 0.5]

    def test_empty_uids_rejected(self):
        with pytest.raises(ValidationError, match="uids list must not be empty"):
            WeightRequest(uids=[], weights=[], version=1)

    def test_empty_weights_rejected(self):
        with pytest.raises(ValidationError, match="weights list must not be empty"):
            WeightRequest(uids=[0], weights=[], version=1)

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValidationError, match="same length"):
            WeightRequest(uids=[0, 1, 2], weights=[0.5, 0.5], version=1)

    def test_duplicate_uids_rejected(self):
        with pytest.raises(ValidationError, match="duplicate"):
            WeightRequest(uids=[0, 1, 0], weights=[0.3, 0.3, 0.4], version=1)

    def test_negative_uid_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            WeightRequest(uids=[-1, 0], weights=[0.5, 0.5], version=1)

    def test_negative_version_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            WeightRequest(uids=[0], weights=[1.0], version=-1)

    def test_nan_weight_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            WeightRequest(uids=[0], weights=[float("nan")], version=1)

    def test_inf_weight_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            WeightRequest(uids=[0], weights=[float("inf")], version=1)

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            WeightRequest(uids=[0], weights=[-0.5], version=1)

    def test_max_uids_limit(self):
        """Request at exactly MAX_UIDS should succeed."""
        uids = list(range(MAX_UIDS))
        weights = [1.0] * MAX_UIDS
        req = WeightRequest(uids=uids, weights=weights, version=1)
        assert len(req.uids) == MAX_UIDS

    def test_exceeds_max_uids_rejected(self):
        uids = list(range(MAX_UIDS + 1))
        weights = [1.0] * (MAX_UIDS + 1)
        with pytest.raises(ValidationError, match="exceeds maximum"):
            WeightRequest(uids=uids, weights=weights, version=1)

    def test_single_uid_valid(self):
        req = WeightRequest(uids=[42], weights=[1.0], version=0)
        assert req.uids == [42]

    def test_large_uid_valid(self):
        req = WeightRequest(uids=[65535], weights=[1.0], version=1)
        assert req.uids == [65535]

    def test_zero_weight_valid(self):
        req = WeightRequest(uids=[0], weights=[0], version=1)
        assert req.weights == [0]
