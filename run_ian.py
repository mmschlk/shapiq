import numpy as np

# from ian_code import ShapleySGD
from shapiq import KernelSHAPIQ

if __name__ == "__main__":

    # n_players = 10
    # weights = np.array([0.1 * i for i in range(n_players)])
    #
    # def linear_model(x: np.ndarray) -> np.ndarray:
    #     """Takes a matrix of shape (n_samples, n_features) and returns a vector of shape
    #     (n_samples,)."""
    #     return np.dot(x, weights)
    #
    # def model_expand_dims(x: np.ndarray) -> np.ndarray:
    #     x_arr = np.array(x)
    #     out = linear_model(x_arr)
    #     if len(out.shape) == 2:
    #         out = out[:, :, None]
    #     return out
    #
    # result_dict = ShapleySGD(
    #     surrogate=model_expand_dims,
    #     d=n_players,
    #     num_subsets=10_000,
    #     mbsize=1_000,
    #     step=0.001,
    #     step_type="constant",
    #     sampling="importance",
    #     averaging="uniform",
    #     return_interval=128,
    #     C=1,
    #     phi_0=False,
    # )
    # print(result_dict)
    #

    n_players = 12
    order = 2

    # setup model
    weights = np.array([0.1 * i for i in range(n_players)])
    interactions = [(i, i + 1) for i in range(n_players - 1)]

    def linear_model(x: np.ndarray) -> np.ndarray:
        """Takes a matrix of shape (n_samples, n_features) and returns a vector of shape
        (n_samples,)."""
        out = np.dot(x, weights)
        for interaction in interactions:
            out += np.prod([x[:, i] for i in interaction], axis=0)
        return out

    # compute with exact fsii
    if n_players < 13:
        from shapiq import ExactComputer

        computer = ExactComputer(n_players=n_players, game=linear_model, evaluate_game=True)
        fsii = computer(index="FSII", order=order)
        print(fsii)

    # compute with regression fsii
    two_fsii = KernelSHAPIQ(
        index="k-SII",
        max_order=order,
        n=n_players,
    )
    fsii = two_fsii.approximate(2**n_players, linear_model)
    print(fsii)
