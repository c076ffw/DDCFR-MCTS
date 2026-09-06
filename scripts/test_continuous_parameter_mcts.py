import math

from ddcfr.cfr import DDCFRSolver
from ddcfr.game.game_config import KuhnPoker
from ddcfr.mcts.continuous_parameter_mcts import (
    ContinuousParameterMCTS,
)
from ddcfr.utils.logger import Logger


INITIAL_ITERATIONS = 10


def main():

    game_config = KuhnPoker(
        iterations=100
    )

    logger = Logger(
        writer_strings=[]
    )

    solver = DDCFRSolver(
        game_config,
        logger,
    )

    # =====================================
    # iteration 10まで固定parameterで進める
    # =====================================

    for _ in range(INITIAL_ITERATIONS):

        solver.num_iterations += 1

        solver.iteration(
            alpha=1.5,
            beta=0.0,
            gamma=2.0,
        )

    before_iteration = solver.num_iterations
    before_conv = solver.calc_conv()

    print("===================================")
    print("CURRENT STATE")
    print("===================================")

    print(
        f"iteration      : {before_iteration}"
    )

    print(
        f"exploitability : {before_conv}"
    )

    # =====================================
    # Continuous MCTS
    # =====================================

    mcts = ContinuousParameterMCTS(
        simulations=100,
        max_depth=3,

        exploration_constant=math.sqrt(2),

        # Progressive Widening
        pw_k=2.0,
        pw_alpha=0.5,

        # parameter space
        alpha_range=(0.0, 5.0),
        beta_range=(-5.0, 0.0),
        gamma_range=(0.0, 5.0),

        # 今は固定
        tau=5,

        seed=0,
    )

    best_action, stats = mcts.search(
        solver
    )

    # =====================================
    # 結果表示
    # =====================================

    print("\n===================================")
    print("CONTINUOUS MCTS RESULT")
    print("===================================")

    print(
        f"generated root actions : "
        f"{len(stats)}"
    )

    print()

    for i, stat in enumerate(
        stats,
        start=1,
    ):

        alpha, beta, gamma, tau = (
            stat["action"]
        )

        print(
            f"{i:2d}. "
            f"alpha={alpha:.4f} "
            f"beta={beta:.4f} "
            f"gamma={gamma:.4f} "
            f"tau={tau} "
            f"visits={stat['visits']:3d} "
            f"value={stat['mean_value']:+.6f}"
        )

    # =====================================
    # 選択されたparameter
    # =====================================

    print("\n===================================")
    print("SELECTED ACTION")
    print("===================================")

    alpha, beta, gamma, tau = best_action

    print(f"alpha = {alpha}")
    print(f"beta  = {beta}")
    print(f"gamma = {gamma}")
    print(f"tau   = {tau}")

    # =====================================
    # 本物solver確認
    # =====================================

    after_iteration = solver.num_iterations
    after_conv = solver.calc_conv()

    print("\n===================================")
    print("REAL SOLVER CHECK")
    print("===================================")

    print(
        f"iteration      : {after_iteration}"
    )

    print(
        f"exploitability : {after_conv}"
    )

    print(
        "iteration unchanged :",
        before_iteration == after_iteration,
    )

    print(
        "conv unchanged      :",
        abs(
            before_conv - after_conv
        ) < 1e-12,
    )

    logger.close()


if __name__ == "__main__":
    main()