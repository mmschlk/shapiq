from __future__ import annotations

import pytest

from shapiq.explainer.nn.threshold_nn import ThresholdNNExplainer


def test_invalid_radius(sklearn_tnn_model) -> None:
    sklearn_tnn_model.radius = "Not a number"
    with pytest.raises(TypeError, match="Expected .*radius to be int or float"):
        ThresholdNNExplainer(sklearn_tnn_model)
