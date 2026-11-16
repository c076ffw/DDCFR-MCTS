import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from ddcfr.cfr import DDCFRSolver
from ddcfr.game.game_config import KuhnPoker
from ddcfr.mcts.full_horizon_mcts import FullHorizonParameterMCTS
from ddcfr.mcts.simulator import apply_action
from ddcfr.utils.convergence import ConvergenceRecorder
from ddcfr.utils.logger import Logger


TOTAL_ITERATIONS = 1000
SIMULATIONS = 12
MAX_DEPTH = 5
TAU = 20


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Choose each DDCFR parameter block using terminal MCTS rollouts."
    )
    parser.add_argument("--iterations", type=positive_int, default=TOTAL_ITERATIONS)
    parser.add_argument("--simulations", type=positive_int, default=SIMULATIONS)
    parser.add_argument("--max-depth", type=positive_int, default=MAX_DEPTH)
    parser.add_argument("--tau", type=positive_int, default=TAU)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-interval",
        type=positive_int,
        default=1,
        help="Record actual exploitability every N iterations, plus each block end",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory; default: results/ContinuousMCTS/KuhnPoker/<UTC timestamp>",
    )
    parser.add_argument(
        "--simulation-budget",
        type=nonnegative_int,
        default=None,
        help="DDCFR iteration budget per search; 0 is unlimited; default: "
        "max(iterations, simulations * max_depth * tau)",
    )
    args = parser.parse_args(argv)
    if args.simulation_budget is None:
        simulation_budget = max(
            args.iterations, args.simulations * args.max_depth * args.tau
        )
    elif args.simulation_budget == 0:
        simulation_budget = None
    else:
        simulation_budget = args.simulation_budget
        if simulation_budget < args.iterations:
            parser.error(
                "--simulation-budget must be at least --iterations so that one "
                "rollout can reach the final iteration; use 0 for unlimited"
            )

    output_dir = args.output_dir
    if output_dir is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_dir = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "ContinuousMCTS"
            / "KuhnPoker"
            / run_name
        )
    output_dir = output_dir.resolve()
    try:
        recorder = ConvergenceRecorder(output_dir, "KuhnPoker - Full-horizon MCTS")
    except FileExistsError:
        parser.error(
            f"{output_dir / 'convergence.csv'} already exists; "
            "choose a new --output-dir"
        )

    logger = Logger(writer_strings=[])
    try:
        game_config = KuhnPoker(iterations=args.iterations)
        solver = DDCFRSolver(game_config, logger)
        mcts = FullHorizonParameterMCTS(
            total_iterations=args.iterations,
            simulations=args.simulations,
            max_depth=args.max_depth,
            exploration_constant=math.sqrt(2),
            pw_k=2.0,
            pw_alpha=0.5,
            alpha_range=(1.0, 2.0),
            beta_range=(-1.0, 0.0),
            gamma_range=(1.0, 3.0),
            tau=args.tau,
            seed=args.seed,
            simulation_budget=simulation_budget,
        )

        print("===================================")
        print("FULL-HORIZON MCTS PARAMETER SEARCH")
        print("===================================")
        print(f"iterations  : {args.iterations}")
        print(f"simulations : {args.simulations}")
        print(f"tree depth  : {args.max_depth}")
        print(f"tau         : {args.tau}")
        print(f"seed        : {args.seed}")
        print(f"eval interval: {args.eval_interval}")
        print(f"results     : {output_dir}")
        print(
            "search budget: "
            f"{simulation_budget if simulation_budget is not None else 'unlimited'} "
            "simulated iterations"
        )

        total_start = time.perf_counter()
        total_mcts_time = 0.0
        search_history = []
        selected_history = []
        recorder.record(0, solver.calc_conv(), time.perf_counter() - total_start)
        recorder.save_plot()

        while solver.num_iterations < args.iterations:
            iteration = solver.num_iterations
            search_start = time.perf_counter()
            # 各探索のrolloutは、現在位置によらず同じ最終イテレーションまで進む。
            action, stats = mcts.search(solver)
            search_time = time.perf_counter() - search_start
            total_mcts_time += search_time
            search_history.append(stats)

            alpha, beta, gamma, tau = action
            conv_before = stats["initial_conv"]
            print("\n===================================")
            print(f"iteration={iteration:4d} conv={conv_before:.10f}")
            print(
                f"selected iterations {iteration + 1:4d}-{iteration + tau:4d}: "
                f"alpha={alpha:.4f} beta={beta:.4f} "
                f"gamma={gamma:.4f} tau={tau}"
            )
            print(
                f"completed simulations={stats['simulations']} "
                f"simulated iterations={stats['simulated_iterations']}"
            )
            print(
                "best rollout final exploitability="
                f"{stats['best_conv']:.10f}"
            )
            print(f"MCTS time={search_time:.4f} sec")

            # 次のブロックのみ本物のsolverに適用し、その状態から再探索する。
            # 仮実行の中間評価は増やさず、実測した収束度だけを記録する。
            block_end = solver.num_iterations + tau
            for _ in range(tau):
                apply_action(solver, (alpha, beta, gamma, 1))
                if (
                    solver.num_iterations % args.eval_interval == 0
                    or solver.num_iterations == block_end
                ):
                    recorder.record(
                        solver.num_iterations,
                        solver.calc_conv(),
                        time.perf_counter() - total_start,
                        action,
                    )
            conv_after = recorder.history[-1]["exploitability"]
            recorder.save_plot()
            print(f"actual conv={conv_after:.10f}")
            selected_history.append(
                {
                    "iteration": iteration,
                    "end_iteration": solver.num_iterations,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "tau": tau,
                    "conv_before": conv_before,
                    "conv_after": conv_after,
                    "simulations": stats["simulations"],
                    "simulated_iterations": stats["simulated_iterations"],
                    "terminal_rollout_conv": stats["best_conv"],
                }
            )

        final_conv = selected_history[-1]["conv_after"]
        total_time = time.perf_counter() - total_start

        print("\n===================================")
        print("FINISHED")
        print("===================================")
        print(f"iterations                     : {solver.num_iterations}")
        print(f"actual final exploitability    : {final_conv:.10f}")
        print(f"total time                     : {total_time:.4f} sec")
        print(f"total MCTS time                : {total_mcts_time:.4f} sec")
        print(f"convergence CSV                : {recorder.csv_path}")
        print(f"convergence graph              : {recorder.plot_path}")

        print("\n===================================")
        print("ACTUAL PARAMETER HISTORY")
        print("===================================")
        for item in selected_history:
            print(
                f"{item['iteration']:4d} -> {item['end_iteration']:4d}: "
                f"alpha={item['alpha']:.4f} "
                f"beta={item['beta']:.4f} "
                f"gamma={item['gamma']:.4f} "
                f"tau={item['tau']} "
                f"conv: {item['conv_before']:.8f} -> {item['conv_after']:.8f}"
            )

        return {
            "stats": search_history,
            "selected_history": selected_history,
            "final_conv": final_conv,
            "iterations": solver.num_iterations,
            "convergence_history": recorder.history,
            "output_dir": output_dir,
        }
    finally:
        try:
            recorder.close()
        finally:
            logger.close()


if __name__ == "__main__":
    main()
