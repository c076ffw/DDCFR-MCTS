from ddcfr.cfr import DDCFRSolver
from ddcfr.game.game_config import KuhnPoker
from ddcfr.mcts.simulator import simulate_action
from ddcfr.utils.logger import Logger


TOTAL_ITERATIONS = 100


ACTIONS = [
    (1.5,  0.0, 2.0, 5),
    (1.0,  0.0, 2.0, 5),
    (2.0,  0.0, 2.0, 5),
    (3.0, -2.0, 4.0, 5),
    (1.5, -1.0, 3.0, 5),
]


def select_best_action(solver):

    results = []

    for action in ACTIONS:

        _, conv_before, conv_after = simulate_action(
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

    return min(
        results,
        key=lambda x: x["conv_after"],
    )


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

    print("===================================")
    print("GREEDY PARAMETER SEARCH")
    print("===================================")

    while solver.num_iterations < TOTAL_ITERATIONS:

        current_conv = solver.calc_conv()

        # ---------------------------------
        # 現在状態から最良parameterを探索
        # ---------------------------------

        best_result = select_best_action(solver)

        alpha, beta, gamma, tau = best_result["action"]

        print(
            f"\niteration={solver.num_iterations:3d} "
            f"conv={current_conv:.10f}"
        )

        print(
            f"selected="
            f"({alpha}, {beta}, {gamma}, {tau})"
        )

        print(
            f"predicted conv="
            f"{best_result['conv_after']:.10f}"
        )

        # ---------------------------------
        # 選ばれたparameterを
        # 今度は本物のsolverに適用
        # ---------------------------------

        for _ in range(tau):

            if solver.num_iterations >= TOTAL_ITERATIONS:
                break

            solver.num_iterations += 1

            solver.iteration(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

        actual_conv = solver.calc_conv()

        print(
            f"actual conv="
            f"{actual_conv:.10f}"
        )

    final_conv = solver.calc_conv()

    print("\n===================================")
    print("FINISHED")
    print("===================================")
    print(f"iterations           : {solver.num_iterations}")
    print(f"final exploitability : {final_conv}")

    logger.close()


if __name__ == "__main__":
    main()