from pathlib import Path

from ddcfr.cfr import DDCFRSolver
from ddcfr.game.game_config import KuhnPoker
from ddcfr.utils.logger import Logger


# =========================
# 実験設定
# =========================

TOTAL_ITERATIONS = 100

ALPHA = 1.5
BETA = 0.0
GAMMA = 2.0
TAU = 5


def main():

    # Kuhn Pokerを使用
    game_config = KuhnPoker(iterations=TOTAL_ITERATIONS)

    # 結果保存先
    folder = (
        Path(__file__).absolute().parents[1]
        / "results"
        / "FixedDDCFR"
        / game_config.name
    )

    folder.mkdir(exist_ok=True, parents=True)

    # CSVに結果を保存
    logger = Logger(
        writer_strings=["csv"],
        folder=folder,
    )

    # DDCFR Solverを作成
    solver = DDCFRSolver(game_config, logger)

    # iteration 0 の exploitability を計算
    solver.after_iteration(
        "iterations",
        eval_iterations_interval=1,
    )

    print("===================================")
    print("Fixed-parameter DDCFR")
    print("===================================")
    print(f"Game  : {game_config.name}")
    print(f"alpha : {ALPHA}")
    print(f"beta  : {BETA}")
    print(f"gamma : {GAMMA}")
    print(f"tau   : {TAU}")
    print("===================================")

    # DDCFR実行
    while solver.num_iterations < TOTAL_ITERATIONS:

        print(
            f"\niteration {solver.num_iterations}: "
            f"use ({ALPHA}, {BETA}, {GAMMA}, {TAU})"
        )

        # tau iterationの間、同じparameterを使用
        for _ in range(TAU):

            if solver.num_iterations >= TOTAL_ITERATIONS:
                break

            solver.num_iterations += 1

            solver.iteration(
                alpha=ALPHA,
                beta=BETA,
                gamma=GAMMA,
            )

            solver.after_iteration(
                "iterations",
                eval_iterations_interval=1,
            )

        print(
            f"iteration = {solver.num_iterations}, "
            f"exploitability = {solver.conv_history[-1]}"
        )

    logger.close()

    print("\nFinished.")
    print(f"Final exploitability: {solver.conv_history[-1]}")
    print(f"Results: {folder}")


if __name__ == "__main__":
    main()