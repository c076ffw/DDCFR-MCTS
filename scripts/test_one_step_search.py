from ddcfr.cfr import DDCFRSolver
from ddcfr.game.game_config import KuhnPoker
from ddcfr.mcts.simulator import simulate_action
from ddcfr.utils.logger import Logger


INITIAL_ITERATIONS = 10

BASE_ALPHA = 1.5
BASE_BETA = 0.0
BASE_GAMMA = 2.0


ACTIONS = [
    (1.5,  0.0, 2.0, 5),
    (1.0,  0.0, 2.0, 5),
    (2.0,  0.0, 2.0, 5),
    (3.0, -2.0, 4.0, 5),
    (1.5, -1.0, 3.0, 5),
]


def main():

    game_config = KuhnPoker(iterations=100)

    logger = Logger(writer_strings=[])

    solver = DDCFRSolver(
        game_config,
        logger,
    )

    # =====================================
    # まず本物のsolverを10 iteration進める
    # =====================================

    for _ in range(INITIAL_ITERATIONS):

        solver.num_iterations += 1

        solver.iteration(
            alpha=BASE_ALPHA,
            beta=BASE_BETA,
            gamma=BASE_GAMMA,
        )

    current_conv = solver.calc_conv()

    print("===================================")
    print("CURRENT STATE")
    print("===================================")
    print(f"iteration      : {solver.num_iterations}")
    print(f"exploitability : {current_conv}")

    print("\n===================================")
    print("TRY ALL ACTIONS")
    print("===================================")

    results = []

    # =====================================
    # 全候補を仮実行
    # =====================================

    for action in ACTIONS:

        sim_solver, conv_before, conv_after = simulate_action(
            solver,
            action,
        )

        improvement = conv_before - conv_after

        results.append(
            {
                "action": action,
                "conv_after": conv_after,
                "improvement": improvement,
            }
        )

        print(
            f"action={action} "
            f"-> exploitability={conv_after:.10f} "
            f"improvement={improvement:+.10f}"
        )

    # =====================================
    # exploitabilityが一番小さい候補を選択
    # =====================================

    best_result = min(
        results,
        key=lambda x: x["conv_after"],
    )

    best_action = best_result["action"]

    print("\n===================================")
    print("BEST ACTION")
    print("===================================")
    print(f"action         : {best_action}")
    print(
        f"exploitability : "
        f"{best_result['conv_after']}"
    )
    print(
        f"improvement    : "
        f"{best_result['improvement']}"
    )

    # 本物が変わっていないことも確認
    print("\n===================================")
    print("REAL SOLVER CHECK")
    print("===================================")
    print(f"iteration      : {solver.num_iterations}")
    print(f"exploitability : {solver.calc_conv()}")

    logger.close()


if __name__ == "__main__":
    main()