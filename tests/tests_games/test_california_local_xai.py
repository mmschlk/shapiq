"""This test module contains all tests regarding the CalifroniaHousingGame."""
import pytest

from shapiq.games.tabular import CaliforniaHousing


@pytest.mark.parametrize("model", ["torch_nn", "sklearn_gbt", "invalid"])
def test_basic_function(model):
    """Tests the CaliforniaHousing game with a small regression dataset."""

    if model == "invalid":
        with pytest.raises(ValueError):
            _ = CaliforniaHousing(model=model)
        return

    x_explain = 0
    if model == "torch_nn":  # test here the auto select
        x_explain = None

    game = CaliforniaHousing(x_explain=x_explain, model=model)
    game.precompute()
    assert game.n_players == 8
    assert len(game.feature_names) == 8
