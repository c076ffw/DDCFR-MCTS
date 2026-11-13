"""Train DDCFR and DDCFR+MCTS independently, then play heads-up Kuhn Poker.

Run from the repository root with the existing DDCFR conda environment:
    python scripts/play_ddcfr_vs_mcts.py --iterations 20 --hands 100
See scripts/README_ddcfr_vs_mcts.md for options and result interpretation.
"""

import argparse
import csv
import json
import math
import random
import secrets
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddcfr.cfr import DDCFRSolver
from ddcfr.game.game_config import KuhnPoker
from ddcfr.mcts.full_horizon_mcts import FullHorizonParameterMCTS
from ddcfr.mcts.simulator import apply_action
from ddcfr.utils.logger import Logger


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


def exact_returns(state, policies):
    """Integrate every deal and mixed action for the given seat policies."""
    if state.is_terminal():
        return list(state.returns())
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
    else:
        outcomes = policies[state.current_player()](state).items()
    result = [0.0, 0.0]
    for action, probability in outcomes:
        if probability == 0:
            continue
        child_returns = exact_returns(state.child(action), policies)
        for seat in range(2):
            result[seat] += probability * child_returns[seat]
    return result


def evaluate_exact(game, ddcfr_policy, mcts_policy):
    """Calculate head-to-head chip expectations with equal weight for both seats."""
    forward = exact_returns(game.new_initial_state(), [ddcfr_policy, mcts_policy])
    reverse = exact_returns(game.new_initial_state(), [mcts_policy, ddcfr_policy])
    ddcfr_mean = (forward[0] + reverse[1]) / 2
    mcts_mean = (forward[1] + reverse[0]) / 2
    return {
        "ddcfr_as_player0": forward[0],
        "ddcfr_as_player1": reverse[1],
        "ddcfr_mean_return": ddcfr_mean,
        "mcts_mean_return": mcts_mean,
    }


def play_hand(game, policies, action_rng, chance_seed):
    """Deal cards and sample both players' actions until the hand is terminal."""
    chance_rng = random.Random(chance_seed)
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            rng = chance_rng
        else:
            outcomes = list(policies[state.current_player()](state).items())
            rng = action_rng
        actions, probabilities = zip(*outcomes)
        action = rng.choices(actions, weights=probabilities, k=1)[0]
        state.apply_action(action)
    return list(state.returns())


def evaluate_matches(game, ddcfr_policy, mcts_policy, hands=100, seed=None):
    """Play balanced hands with fresh randomness unless a match seed is given."""
    if hands < 2 or hands % 2:
        raise ValueError("hands must be an even integer >= 2")
    if seed is None:
        seed = secrets.randbits(128)
    action_rng = random.Random(seed)
    deal_rng = random.Random(seed ^ 0xDDCF)
    records = []
    for hand in range(hands):
        ddcfr_seat = hand % 2
        chance_seed = deal_rng.getrandbits(64)
        policies = (
            [ddcfr_policy, mcts_policy]
            if ddcfr_seat == 0 else [mcts_policy, ddcfr_policy]
        )
        returns = play_hand(game, policies, action_rng, chance_seed)
        if returns[ddcfr_seat] > returns[1 - ddcfr_seat]:
            winner = "DDCFR"
        elif returns[ddcfr_seat] < returns[1 - ddcfr_seat]:
            winner = "DDCFR+MCTS"
        else:
            winner = "draw"
        records.append({"hand": hand + 1, "ddcfr_seat": ddcfr_seat, "winner": winner})

    def count_results(rows):
        return {
            "ddcfr_wins": sum(row["winner"] == "DDCFR" for row in rows),
            "mcts_wins": sum(row["winner"] == "DDCFR+MCTS" for row in rows),
            "draws": sum(row["winner"] == "draw" for row in rows),
        }

    summary = {
        "hands": hands,
        **count_results(records),
        "by_seat": {},
    }
    for seat, name in ((0, "first"), (1, "second")):
        seat_records = [row for row in records if row["ddcfr_seat"] == seat]
        summary["by_seat"][name] = {"games": len(seat_records), **count_results(seat_records)}
    return summary, records


def print_matches(summary):
    print(f"対戦結果（{summary['hands']}戦）")
    print(f"DDCFR      : {summary['ddcfr_wins']}勝 {summary['mcts_wins']}敗")
    print(f"DDCFR+MCTS : {summary['mcts_wins']}勝 {summary['ddcfr_wins']}敗")
    print(f"引き分け   : {summary['draws']}戦")
    for name, label in (("first", "先手"), ("second", "後手")):
        counts = summary["by_seat"][name]
        print(f"DDCFRが{label}（{counts['games']}戦）: "
              f"{counts['ddcfr_wins']}勝 {counts['mcts_wins']}敗 {counts['draws']}分")


def print_metrics(exact, training):
    print("\n厳密な対戦期待値（チップ/戦、先手・後手を半分ずつ）")
    print(f"DDCFR      : {exact['ddcfr_mean_return']:+.10g}")
    print(f"DDCFR+MCTS : {exact['mcts_mean_return']:+.10g}")
    print("\nExploitability（小さいほど戦略の隙が小さい）")
    print(f"DDCFR      : {training['ddcfr']['exploitability']:.10g}")
    print(f"DDCFR+MCTS : {training['mcts']['exploitability']:.10g}")


def training_row(solver, start_iteration, action, before, after, stats=None):
    alpha, beta, gamma, tau = action
    stats = stats or {}
    return {
        "start_iteration": start_iteration,
        "end_iteration": solver.num_iterations,
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "tau": int(tau),
        "conv_before": float(before),
        "conv_after": float(after),
        "simulations": stats.get("simulations", 0),
        "simulated_iterations": stats.get("simulated_iterations", 0),
    }


def train_ddcfr(args, logger):
    start_time = time.perf_counter()
    game_config = KuhnPoker(iterations=args.iterations)
    history = []
    if args.baseline == "learned":
        import torch
        from ddcfr.cfr.cfr_env import CFREnv
        from ddcfr.es.cfr_model import CFRModel

        env = CFREnv(game_config, logger, args.iterations, seed=args.seed)
        model = CFRModel(env.observation_space, env.action_space, torch.device("cpu"))
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        model.eval()
        observation = env.reset()
        solver = env.solver
        done = False
        while not done:
            start_iteration = solver.num_iterations
            before = solver.conv_history[-1]
            action = model(observation)
            alpha, beta, gamma, tau = env._unscale_action(action)
            tau = min(tau, args.iterations - start_iteration)
            observation, _, done, _ = env.step(action)
            history.append(training_row(
                solver, start_iteration, (alpha, beta, gamma, tau),
                before, solver.conv_history[-1],
            ))
    else:
        solver = DDCFRSolver(game_config, logger)
        before = solver.calc_conv()
        while solver.num_iterations < args.iterations:
            start_iteration = solver.num_iterations
            tau = min(args.tau, args.iterations - start_iteration)
            action = (1.5, 0.0, 2.0, tau)
            apply_action(solver, action)
            after = solver.calc_conv()
            history.append(training_row(solver, start_iteration, action, before, after))
            before = after
    return solver, {
        "iterations": solver.num_iterations,
        "exploitability": history[-1]["conv_after"],
        "seconds": time.perf_counter() - start_time,
        "history": history,
    }


def train_mcts(args, logger):
    start_time = time.perf_counter()
    solver = DDCFRSolver(KuhnPoker(iterations=args.iterations), logger)
    mcts = FullHorizonParameterMCTS(
        total_iterations=args.iterations,
        simulations=args.simulations,
        max_depth=args.max_depth,
        tau=args.tau,
        seed=args.mcts_seed,
        alpha_range=tuple(args.alpha_range),
        beta_range=tuple(args.beta_range),
        gamma_range=tuple(args.gamma_range),
        simulation_budget=args.simulation_budget,
    )
    history = []
    while solver.num_iterations < args.iterations:
        start_iteration = solver.num_iterations
        action, stats = mcts.search(solver)
        apply_action(solver, action)
        after = solver.calc_conv()
        history.append(training_row(
            solver, start_iteration, action, stats["initial_conv"], after, stats,
        ))
    return solver, {
        "iterations": solver.num_iterations,
        "exploitability": history[-1]["conv_after"],
        "seconds": time.perf_counter() - start_time,
        "simulated_iterations": sum(row["simulated_iterations"] for row in history),
        "history": history,
    }


def save_results(folder, result, records):
    # A new directory prevents overwriting any existing experiment or source.
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (folder / "matches.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (folder / "training.csv").open("x", encoding="utf-8", newline="") as stream:
        fields = ["agent"] + list(result["training"]["ddcfr"]["history"][0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for agent, training in result["training"].items():
            for row in training["history"]:
                writer.writerow(dict(agent=agent, **row))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Play DDCFR against DDCFR+MCTS in Kuhn Poker, swapping seats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=positive_int, default=1000)
    parser.add_argument("--simulations", type=positive_int, default=12)
    parser.add_argument("--max-depth", type=positive_int, default=5)
    parser.add_argument("--tau", type=positive_int, default=20)
    parser.add_argument("--seed", type=nonnegative_int, default=0,
                        help="training seed for the DDCFR baseline")
    parser.add_argument("--mcts-seed", type=nonnegative_int, default=None,
                        help="MCTS search seed; omit for fresh randomness on each run")
    parser.add_argument("--match-seed", type=nonnegative_int, default=None,
                        help="match seed for reproducible deals and actions; "
                        "omit for fresh randomness on each run")
    parser.add_argument("--hands", type=positive_int, default=10000, help="total games; even and >= 2")
    parser.add_argument("--baseline", choices=["learned", "fixed"], default="learned",
                        help="learned DDCFR or fixed parameters (1.5, 0, 2)")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "model.pkl",
                        help="ES checkpoint used by the learned DDCFR baseline")
    parser.add_argument("--alpha-range", type=float, nargs=2, default=(1.0, 2.0), metavar=("MIN", "MAX"))
    parser.add_argument("--beta-range", type=float, nargs=2, default=(-1.0, 0.0), metavar=("MIN", "MAX"))
    parser.add_argument("--gamma-range", type=float, nargs=2, default=(1.0, 3.0), metavar=("MIN", "MAX"))
    parser.add_argument("--simulation-budget", type=nonnegative_int, default=None,
                        help="simulated iterations per search; 0=unlimited; "
                        "default=max(iterations, simulations * max_depth * tau)")
    parser.add_argument("--output", type=Path, help="save JSON/CSV to a new directory")
    args = parser.parse_args(argv)
    if args.hands < 2 or args.hands % 2:
        parser.error("--hands must be an even integer >= 2")
    if args.baseline == "learned" and not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if args.output is not None and args.output.exists():
        parser.error(f"--output must be a new directory: {args.output}")
    for name in ("alpha_range", "beta_range", "gamma_range"):
        low, high = getattr(args, name)
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            parser.error(f"--{name.replace('_', '-')} must have finite MIN <= MAX")
    if args.simulation_budget is None:
        args.simulation_budget = max(
            args.iterations, args.simulations * args.max_depth * args.tau,
        )
    elif args.simulation_budget == 0:
        args.simulation_budget = None
    elif args.simulation_budget < args.iterations:
        parser.error("--simulation-budget must be >= --iterations, or 0 for unlimited")

    logger = Logger(writer_strings=[])
    try:
        ddcfr, ddcfr_training = train_ddcfr(args, logger)
        if args.mcts_seed is None:
            args.mcts_seed = secrets.randbits(128)
        mcts, mcts_training = train_mcts(args, logger)
        ddcfr_policy = ddcfr.average_policy()
        mcts_policy = mcts.average_policy()
        exact = evaluate_exact(ddcfr.game, ddcfr_policy, mcts_policy)
        if args.match_seed is None:
            args.match_seed = secrets.randbits(128)
        matches, records = evaluate_matches(
            ddcfr.game, ddcfr_policy, mcts_policy,
            hands=args.hands, seed=args.match_seed,
        )
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        config["game"] = "KuhnPoker"
        result = {
            "config": config,
            "training": {"ddcfr": ddcfr_training, "mcts": mcts_training},
            "matches": matches,
            "exact": exact,
        }
        print_matches(matches)
        print_metrics(exact, result["training"])
        if args.output is not None:
            save_results(args.output, result, records)
        return result
    finally:
        logger.close()


if __name__ == "__main__":
    main()
