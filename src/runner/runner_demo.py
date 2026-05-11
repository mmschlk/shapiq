from benchmark_runner import run_benchmark
from shapiq.approximator import ProxySHAP


def main() -> None:
    run_benchmark(
        game_seed=42,
        max_order=2,
        number_of_different_approx_seeds=5,
        budget=100,
        n_players=10,
        n_basis_games=20,
        min_interaction_size=1,
        index="SII",
        approximator_class=ProxySHAP,
    )


if __name__ == "__main__":
    main()