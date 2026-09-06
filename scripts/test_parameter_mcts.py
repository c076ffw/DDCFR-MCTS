import math

from ddcfr.cfr import DDCFRSolver
from ddcfr.game.game_config import KuhnPoker
from ddcfr.mcts.parameter_mcts import ParameterMCTS
from ddcfr.utils.logger import Logger


INITIAL_ITERATIONS = 10


ACTIONS = [
    (1.5, 0.0, 2.0, 5),
    (1.0, 0.0, 2.0, 5),
    (2.0, 0.0, 2.0, 5),
    (3.0, -2.0, 4.0, 5),
    (1.5, -1.0, 3.0, 5),
]


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
    # iteration 10までDCFRで進める
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
    # MCTS
    # =====================================

    mcts = ParameterMCTS(
        actions=ACTIONS,
        simulations=50,
        max_depth=3,
        exploration_constant=math.sqrt(2),
        seed=0,
    )

    best_action, stats = mcts.search(
        solver
    )

    print("\n===================================")
    print("MCTS RESULT")
    print("===================================")

    for stat in stats:

        print(
            f"action={stat['action']} "
            f"visits={stat['visits']:3d} "
            f"mean_reward="
            f"{stat['mean_value']:+.6f}"
        )

    print("\n===================================")
    print("SELECTED ACTION")
    print("===================================")

    print(best_action)

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