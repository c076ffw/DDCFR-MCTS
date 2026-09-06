from ddcfr.cfr import DDCFRSolver
from ddcfr.game.game_config import KuhnPoker
from ddcfr.mcts.simulator import simulate_action
from ddcfr.utils.logger import Logger


INITIAL_ITERATIONS = 10

BASE_ALPHA = 1.5
BASE_BETA = 0.0
BASE_GAMMA = 2.0

TEST_ACTION = (
    3.0,   # alpha
    -2.0,  # beta
    4.0,   # gamma
    5,     # tau
)


def main():

    game_config = KuhnPoker(iterations=100)

    # 今回はテストなのでファイル出力なし
    logger = Logger(writer_strings=[])

    solver = DDCFRSolver(
        game_config,
        logger,
    )

    # ==========================
    # まず本物を10 iteration進める
    # ==========================

    for _ in range(INITIAL_ITERATIONS):
        solver.num_iterations += 1

        solver.iteration(
            alpha=BASE_ALPHA,
            beta=BASE_BETA,
            gamma=BASE_GAMMA,
        )

    real_conv_before = solver.calc_conv()
    real_iteration_before = solver.num_iterations

    print("================================")
    print("REAL SOLVER BEFORE SIMULATION")
    print("================================")
    print(f"iteration      : {real_iteration_before}")
    print(f"exploitability : {real_conv_before}")

    # ==========================
    # コピー上で候補を仮実行
    # ==========================

    sim_solver, sim_conv_before, sim_conv_after = simulate_action(
        solver,
        TEST_ACTION,
    )

    print("\n================================")
    print("SIMULATION")
    print("================================")
    print(f"action         : {TEST_ACTION}")
    print(f"before iter    : {real_iteration_before}")
    print(f"after iter     : {sim_solver.num_iterations}")
    print(f"before conv    : {sim_conv_before}")
    print(f"after conv     : {sim_conv_after}")

    # ==========================
    # 本物が変わってないか確認
    # ==========================

    real_conv_after = solver.calc_conv()

    print("\n================================")
    print("REAL SOLVER AFTER SIMULATION")
    print("================================")
    print(f"iteration      : {solver.num_iterations}")
    print(f"exploitability : {real_conv_after}")

    print("\n================================")
    print("CHECK")
    print("================================")

    iteration_unchanged = (
        solver.num_iterations == real_iteration_before
    )

    conv_unchanged = (
        abs(real_conv_after - real_conv_before) < 1e-12
    )

    print(f"iteration unchanged      : {iteration_unchanged}")
    print(f"exploitability unchanged : {conv_unchanged}")

    if iteration_unchanged and conv_unchanged:
        print("\nSUCCESS: real solver was not modified.")
    else:
        print("\nERROR: real solver was modified.")

    logger.close()


if __name__ == "__main__":
    main()