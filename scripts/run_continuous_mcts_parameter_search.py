import math
import time

from ddcfr.cfr import DDCFRSolver
from ddcfr.game.game_config import KuhnPoker
from ddcfr.mcts.continuous_parameter_mcts import (
    ContinuousParameterMCTS,
)
from ddcfr.utils.logger import Logger


TOTAL_ITERATIONS = 1000

SIMULATIONS = 100
MAX_DEPTH = 3
TAU = 5


def main():

    game_config = KuhnPoker(
        iterations=TOTAL_ITERATIONS
    )

    logger = Logger(
        writer_strings=[]
    )

    solver = DDCFRSolver(
        game_config,
        logger,
    )

    mcts = ContinuousParameterMCTS(
        simulations=SIMULATIONS,
        max_depth=MAX_DEPTH,

        exploration_constant=math.sqrt(2),

        pw_k=2.0,
        pw_alpha=0.5,

        alpha_range=(0.0, 5.0),
        beta_range=(-5.0, 0.0),
        gamma_range=(0.0, 5.0),

        tau=TAU,

        seed=0,
    )

    selected_history = []

    total_start = time.perf_counter()
    total_mcts_time = 0.0

    print("===================================")
    print("CONTINUOUS MCTS PARAMETER SEARCH")
    print("===================================")
    print(f"simulations : {SIMULATIONS}")
    print(f"max depth   : {MAX_DEPTH}")
    print(f"tau         : {TAU}")

    while solver.num_iterations < TOTAL_ITERATIONS:

        iteration = solver.num_iterations
        conv_before = solver.calc_conv()

        # ======================================
        # MCTS
        # ======================================

        search_start = time.perf_counter()

        best_action, stats = mcts.search(
            solver
        )

        search_time = (
            time.perf_counter()
            - search_start
        )

        total_mcts_time += search_time

        alpha, beta, gamma, tau = best_action

        print("\n===================================")

        print(
            f"iteration={iteration:3d} "
            f"conv={conv_before:.10f}"
        )

        print(
            f"generated actions={len(stats)}"
        )

        print(
            "selected="
            f"({alpha:.4f}, "
            f"{beta:.4f}, "
            f"{gamma:.4f}, "
            f"{tau})"
        )

        print(
            f"visits={stats[0]['visits']} "
            f"value={stats[0]['mean_value']:+.6f}"
        )

        print(
            f"MCTS time={search_time:.4f} sec"
        )

        # ======================================
        # 本物のsolverへ適用
        # ======================================

        for _ in range(tau):

            if (
                solver.num_iterations
                >= TOTAL_ITERATIONS
            ):
                break

            solver.num_iterations += 1

            solver.iteration(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

        conv_after = solver.calc_conv()

        print(
            f"actual conv={conv_after:.10f}"
        )

        selected_history.append(
            {
                "iteration": iteration,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "tau": tau,
                "conv_before": conv_before,
                "conv_after": conv_after,
            }
        )

    total_time = (
        time.perf_counter()
        - total_start
    )

    final_conv = solver.calc_conv()

    print("\n===================================")
    print("FINISHED")
    print("===================================")

    print(
        f"iterations           : "
        f"{solver.num_iterations}"
    )

    print(
        f"final exploitability : "
        f"{final_conv}"
    )

    print(
        f"total time           : "
        f"{total_time:.4f} sec"
    )

    print(
        f"total MCTS time      : "
        f"{total_mcts_time:.4f} sec"
    )

    print("\n===================================")
    print("PARAMETER HISTORY")
    print("===================================")

    for item in selected_history:

        print(
            f"{item['iteration']:3d} -> "
            f"alpha={item['alpha']:.4f} "
            f"beta={item['beta']:.4f} "
            f"gamma={item['gamma']:.4f} "
            f"tau={item['tau']} "
            f"conv: "
            f"{item['conv_before']:.8f}"
            f" -> "
            f"{item['conv_after']:.8f}"
        )

    logger.close()


if __name__ == "__main__":
    main()