"""This test module contains all tests regarding the CalifroniaHousingGame."""

import os

import numpy as np
import pytest

from shapiq.games import CaliforniaHousing


@pytest.mark.slow
@pytest.mark.parametrize("model", ["torch_nn", "sklearn_gbt", "invalid"])
def test_basic_function(model):
    """Tests the CaliforniaHousing game with a small regression dataset."""

    if model == "invalid":
        with pytest.raises(ValueError):
            _ = CaliforniaHousing(model=model, x_explain=0)
        return

    x_explain_id = 0
    if model == "torch_nn":  # test here the auto select
        x_explain_id = None

    game = CaliforniaHousing(x_explain=x_explain_id, model=model)
    game.precompute()
    assert game.n_players == 8
    assert len(game.feature_names) == 8
    assert game.n_values_stored == 2**8
    assert game.precomputed

    # test save and load values
    path = f"california_local_xai_{model}_id_{x_explain_id}.npz"
    game.save_values(path)

    assert os.path.exists(path)

    # test init from values file
    new_game = CaliforniaHousing(path_to_values=path)
    assert new_game.n_values_stored == game.n_values_stored
    assert new_game.n_players == game.n_players
    assert new_game.normalize == game.normalize
    assert np.allclose(new_game.value_storage, game.value_storage)

    # clean up
    os.remove(path)
    assert not os.path.exists(path)
