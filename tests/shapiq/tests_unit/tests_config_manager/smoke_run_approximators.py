from __future__ import annotations

import logging

from shapiq.approximator import KernelSHAP, PermutationSamplingSV, StratifiedSamplingSV
from shapiq_games.synthetic.dummy import DummyGame

logger = logging.getLogger(__name__)

def try_run(ApproxClass, name):
    logger.info("--- %s ---", name)
    try:
        n = 4
        game = DummyGame(n)
        # Instantiate with common parameters if supported
        inst = ApproxClass(n=n, random_state=0)
        iv = inst.approximate(budget=20, game=game)
        logger.info(
            "Success: returned InteractionValues, n_players=%s, estimated=%s",
            iv.n_players,
            iv.estimated,
        )
    except TypeError:
        logger.exception("TypeError constructing or calling %s", name)
    except Exception as e:
        logger.exception("Runtime error in %s: %s: %s", name, type(e).__name__, e)

if __name__ == "__main__":
    # Only run a small curated set known to be SV-compatible / commonly used
    try_run(PermutationSamplingSV, "PermutationSamplingSV")
    try_run(StratifiedSamplingSV, "StratifiedSamplingSV")
    try_run(KernelSHAP, "KernelSHAP")
