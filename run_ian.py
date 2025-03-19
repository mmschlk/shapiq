import numpy as np

from ian_code import ShapleySGD

if __name__ == "__main__":

    n_players = 10
    weights = np.array([0.1 * i for i in range(n_players)])

    def linear_model(x: np.ndarray) -> np.ndarray:
        """Takes a matrix of shape (n_samples, n_features) and returns a vector of shape
        (n_samples,)."""
        return np.dot(x, weights)

    def model_expand_dims(x: np.ndarray) -> np.ndarray:
        x_arr = np.array(x)
        out = linear_model(x_arr)
        if len(out.shape) == 2:
            out = out[:, :, None]
        return out

    result_dict = ShapleySGD(
        surrogate=model_expand_dims,
        d=n_players,
        num_subsets=2 ** (n_players + 2),
        mbsize=32,
        step=0.001,
        step_type="constant",
        sampling="importance",
        averaging="uniform",
        return_interval=128,
        C=1,
        phi_0=False,
    )
    print(result_dict)

    from shapiq import ExactComputer

    computer = ExactComputer(n_players=n_players, game=linear_model, evaluate_game=True)
    sv = computer(index="SV", order=1)
    print(sv)
