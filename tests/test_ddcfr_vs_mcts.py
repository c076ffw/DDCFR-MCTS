"""Kuhn Poker match totals, exact matchup returns, and exploitability."""

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

import pyspiel

from scripts import play_ddcfr_vs_mcts as runner


def uniform_policy(state):
    actions = state.legal_actions()
    return {action: 1.0 / len(actions) for action in actions}


def passive_policy(state):
    """Always check or fold in Kuhn Poker."""
    return {state.legal_actions()[0]: 1.0}


def aggressive_policy(state):
    """Always bet or call in Kuhn Poker."""
    return {state.legal_actions()[-1]: 1.0}


class SampledMatchTest(unittest.TestCase):
    def setUp(self):
        self.game = pyspiel.load_game("kuhn_poker")

    def test_passive_player_loses_every_actual_hand_in_both_seats(self):
        # A passive player folds to every bet, so the aggressive player wins
        # regardless of the private cards and which player acts first.
        summary, records = runner.evaluate_matches(
            self.game, passive_policy, aggressive_policy, hands=20, seed=17
        )

        self.assertEqual(summary["hands"], 20)
        self.assertEqual(summary["ddcfr_wins"], 0)
        self.assertEqual(summary["mcts_wins"], 20)
        self.assertEqual(summary["draws"], 0)
        self.assertEqual(len(records), 20)
        self.assertTrue(all(row["winner"] == "DDCFR+MCTS" for row in records))
        for seat in ("first", "second"):
            self.assertEqual(summary["by_seat"][seat], {
                "games": 10, "ddcfr_wins": 0, "mcts_wins": 10, "draws": 0,
            })

    def test_one_hundred_hands_alternate_seats_fifty_times_each(self):
        summary, records = runner.evaluate_matches(
            self.game, uniform_policy, aggressive_policy, hands=100, seed=9
        )

        self.assertEqual([row["hand"] for row in records], list(range(1, 101)))
        self.assertEqual([row["ddcfr_seat"] for row in records], [0, 1] * 50)
        self.assertEqual(summary["by_seat"]["first"]["games"], 50)
        self.assertEqual(summary["by_seat"]["second"]["games"], 50)
        for seat, name in ((0, "first"), (1, "second")):
            seated_records = [row for row in records if row["ddcfr_seat"] == seat]
            for winner, counter in (("DDCFR", "ddcfr_wins"),
                                    ("DDCFR+MCTS", "mcts_wins"),
                                    ("draw", "draws")):
                self.assertEqual(
                    summary["by_seat"][name][counter],
                    sum(row["winner"] == winner for row in seated_records),
                )

    def test_each_hand_is_dealt_independently(self):
        class RecordingGame:
            def __init__(self, game):
                self.game = game
                self.states = []

            def new_initial_state(self):
                state = self.game.new_initial_state()
                self.states.append(state)
                return state

        game = RecordingGame(self.game)
        runner.evaluate_matches(
            game, aggressive_policy, aggressive_policy, hands=100, seed=17
        )

        self.assertEqual(len(game.states), 100)
        self.assertTrue(all(state.is_terminal() for state in game.states))
        deals = [state.history()[:2] for state in game.states]
        # Alternating seats must not force adjacent hands to reuse the cards.
        self.assertTrue(any(deals[i] != deals[i + 1] for i in range(0, 100, 2)))

    def test_seed_reproduces_matches_and_accounting_is_consistent(self):
        first = runner.evaluate_matches(
            self.game, uniform_policy, aggressive_policy, hands=100, seed=123
        )
        second = runner.evaluate_matches(
            self.game, uniform_policy, aggressive_policy, hands=100, seed=123
        )

        self.assertEqual(first, second)
        summary, records = first
        self.assertEqual(
            summary["ddcfr_wins"] + summary["mcts_wins"] + summary["draws"],
            summary["hands"],
        )
        for winner, counter in (("DDCFR", "ddcfr_wins"),
                                ("DDCFR+MCTS", "mcts_wins"), ("draw", "draws")):
            self.assertEqual(
                sum(row["winner"] == winner for row in records), summary[counter]
            )
        self.assertEqual(summary["draws"], 0)
        self.assertEqual(set(summary), {
            "hands", "ddcfr_wins", "mcts_wins", "draws", "by_seat",
        })
        for row in records:
            self.assertEqual(set(row), {"hand", "ddcfr_seat", "winner"})

    def test_omitted_match_seed_draws_fresh_entropy_for_each_evaluation(self):
        expected = [
            runner.evaluate_matches(
                self.game, uniform_policy, aggressive_policy, hands=100, seed=seed
            )
            for seed in (123, 456)
        ]

        with mock.patch.object(runner.secrets, "randbits", side_effect=[123, 456]) as entropy:
            actual = [
                runner.evaluate_matches(
                    self.game, uniform_policy, aggressive_policy, hands=100
                )
                for _ in range(2)
            ]

        self.assertEqual(entropy.call_args_list, [mock.call(128), mock.call(128)])
        self.assertEqual(actual, expected)

    def test_only_positive_even_hand_counts_are_accepted(self):
        for hands in (-2, 0, 1, 3, 99):
            with self.subTest(hands=hands), self.assertRaises(ValueError):
                runner.evaluate_matches(
                    self.game, uniform_policy, uniform_policy, hands=hands
                )
        summary, records = runner.evaluate_matches(
            self.game, passive_policy, aggressive_policy, hands=2
        )
        self.assertEqual(summary["hands"], 2)
        self.assertEqual([row["ddcfr_seat"] for row in records], [0, 1])

    def test_prints_only_totals_without_individual_results_or_payoffs(self):
        summary, _ = runner.evaluate_matches(
            self.game, uniform_policy, aggressive_policy, hands=100, seed=123
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runner.print_matches(summary)

        lines = output.getvalue().splitlines()
        hand_lines = [line for line in lines if re.match(r"第\d+戦\s*\|", line)]
        self.assertEqual(hand_lines, [])
        self.assertEqual(lines, [
            "対戦結果（100戦）",
            f"DDCFR      : {summary['ddcfr_wins']}勝 {summary['mcts_wins']}敗",
            f"DDCFR+MCTS : {summary['mcts_wins']}勝 {summary['ddcfr_wins']}敗",
            f"引き分け   : {summary['draws']}戦",
            "DDCFRが先手（50戦）: "
            f"{summary['by_seat']['first']['ddcfr_wins']}勝 "
            f"{summary['by_seat']['first']['mcts_wins']}敗 0分",
            "DDCFRが後手（50戦）: "
            f"{summary['by_seat']['second']['ddcfr_wins']}勝 "
            f"{summary['by_seat']['second']['mcts_wins']}敗 0分",
        ])
        for absent in ("期待値", "損益", "95%", "CI", "mean_return", "total_return"):
            self.assertNotIn(absent, output.getvalue())


class ExactEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.game = pyspiel.load_game("kuhn_poker")

    def test_uniform_self_play_has_known_seat_returns_and_balanced_value(self):
        state = self.game.new_initial_state()
        returns = runner.exact_returns(state, [uniform_policy, uniform_policy])

        # Uniform random play favors player 0, so averaging the two seats is
        # necessary even when both agents use exactly the same strategy.
        self.assertAlmostEqual(returns[0], 0.125)
        self.assertAlmostEqual(returns[1], -0.125)
        self.assertEqual(state.history(), [])
        exact = runner.evaluate_exact(self.game, uniform_policy, uniform_policy)
        self.assertEqual(set(exact), {
            "ddcfr_as_player0", "ddcfr_as_player1",
            "ddcfr_mean_return", "mcts_mean_return",
        })
        self.assertAlmostEqual(exact["ddcfr_as_player0"], 0.125)
        self.assertAlmostEqual(exact["ddcfr_as_player1"], -0.125)
        self.assertAlmostEqual(exact["ddcfr_mean_return"], 0.0)
        self.assertAlmostEqual(exact["mcts_mean_return"], 0.0)

    def test_aggressive_player_gains_one_chip_in_either_seat(self):
        # A passive player folds to every bet, which fixes the payoff at one
        # chip independently of both players' private cards.
        for ddcfr_policy, mcts_policy, expected in (
            (passive_policy, aggressive_policy, -1.0),
            (aggressive_policy, passive_policy, 1.0),
        ):
            with self.subTest(ddcfr_policy=ddcfr_policy.__name__):
                exact = runner.evaluate_exact(self.game, ddcfr_policy, mcts_policy)
                self.assertAlmostEqual(exact["ddcfr_as_player0"], expected)
                self.assertAlmostEqual(exact["ddcfr_as_player1"], expected)
                self.assertAlmostEqual(exact["ddcfr_mean_return"], expected)
                self.assertAlmostEqual(exact["mcts_mean_return"], -expected)


class RunnerIntegrationTest(unittest.TestCase):
    def test_existing_output_directory_is_rejected_without_changing_contents(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            output = Path(temp_directory)
            sentinel = output / "sentinel.txt"
            sentinel.write_text("existing experiment\n")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    runner.main(["--baseline", "fixed", "--output", str(output)])

            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(sentinel.read_text(), "existing experiment\n")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_short_fixed_baseline_run_writes_new_output(self):
        trained = {}

        def capture_training(name, train):
            def wrapped(*args, **kwargs):
                solver, training = train(*args, **kwargs)
                trained[name] = solver
                return solver, training
            return wrapped

        with tempfile.TemporaryDirectory() as temp_directory:
            output = Path(temp_directory) / "new_match"
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                runner.secrets, "randbits", side_effect=[123456789, 987654321]
            ) as entropy, mock.patch.object(
                runner, "train_ddcfr",
                side_effect=capture_training("ddcfr", runner.train_ddcfr),
            ), mock.patch.object(
                runner, "train_mcts",
                side_effect=capture_training("mcts", runner.train_mcts),
            ):
                result = runner.main(
                    [
                        "--baseline", "fixed",
                        "--iterations", "5",
                        "--simulations", "2",
                        "--max-depth", "1",
                        "--tau", "3",
                        "--hands", "20",
                        "--seed", "7",
                        "--output", str(output),
                    ]
                )

            self.assertEqual(entropy.call_args_list, [mock.call(128), mock.call(128)])
            self.assertEqual(result["config"]["seed"], 7)
            self.assertEqual(result["config"]["mcts_seed"], 123456789)
            self.assertEqual(result["config"]["match_seed"], 987654321)
            self.assertEqual(result["matches"]["hands"], 20)
            for agent in ("ddcfr", "mcts"):
                training = result["training"][agent]
                self.assertEqual(training["iterations"], 5)
                self.assertEqual([row["tau"] for row in training["history"]], [3, 2])
                self.assertEqual(
                    [row["end_iteration"] for row in training["history"]], [3, 5]
                )
                self.assertTrue(math.isfinite(training["exploitability"]))
                self.assertGreater(training["exploitability"], 0.0)
                self.assertAlmostEqual(
                    training["exploitability"], trained[agent].calc_conv()
                )
            self.assertEqual(set(result), {"config", "training", "matches", "exact"})
            self.assertEqual(result["exact"], runner.evaluate_exact(
                trained["ddcfr"].game,
                trained["ddcfr"].average_policy(),
                trained["mcts"].average_policy(),
            ))
            for filename in ("result.json", "matches.csv", "training.csv"):
                self.assertTrue((output / filename).is_file(), filename)
            saved = json.loads((output / "result.json").read_text())
            self.assertEqual(saved["config"]["mcts_seed"], 123456789)
            self.assertEqual(saved["config"]["match_seed"], 987654321)
            self.assertEqual(saved["matches"], result["matches"])
            self.assertEqual(saved["exact"], result["exact"])
            for agent in ("ddcfr", "mcts"):
                self.assertEqual(
                    saved["training"][agent]["exploitability"],
                    result["training"][agent]["exploitability"],
                )
            with (output / "matches.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["hand", "ddcfr_seat", "winner"])
                saved_records = list(reader)
            self.assertEqual(len(saved_records), 20)
            self.assertEqual(
                sum(row["winner"] == "DDCFR" for row in saved_records),
                result["matches"]["ddcfr_wins"],
            )

    def test_explicit_zero_mcts_seed_and_match_seed_reproduce_training_and_matches(self):
        arguments = [
            "--baseline", "fixed",
            "--iterations", "2",
            "--simulations", "1",
            "--max-depth", "1",
            "--tau", "1",
            "--hands", "20",
            "--seed", "7",
            "--mcts-seed", "0",
            "--match-seed", "123",
        ]
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            runner.secrets, "randbits"
        ) as entropy, mock.patch.object(
            runner, "evaluate_matches", wraps=runner.evaluate_matches
        ) as evaluate, mock.patch.object(
            runner, "train_mcts", wraps=runner.train_mcts
        ) as train_mcts:
            first = runner.main(arguments)
            second = runner.main(arguments)

        entropy.assert_not_called()
        self.assertEqual(first["config"]["seed"], 7)
        self.assertEqual(first["config"]["mcts_seed"], 0)
        self.assertEqual(first["config"]["match_seed"], 123)
        self.assertEqual([call.args[0].seed for call in train_mcts.call_args_list], [7, 7])
        self.assertEqual([call.args[0].mcts_seed for call in train_mcts.call_args_list], [0, 0])
        self.assertEqual([call.kwargs["seed"] for call in evaluate.call_args_list], [123, 123])
        self.assertEqual(first["matches"], second["matches"])
        self.assertEqual(first["exact"], second["exact"])
        for agent in ("ddcfr", "mcts"):
            self.assertEqual(
                first["training"][agent]["history"],
                second["training"][agent]["history"],
            )

    def test_omitted_mcts_seed_changes_search_and_saved_seeds_reproduce_it(self):
        arguments = [
            "--baseline", "fixed",
            "--iterations", "4",
            "--simulations", "2",
            "--max-depth", "2",
            "--tau", "1",
            "--hands", "20",
            "--seed", "7",
            "--match-seed", "123",
        ]
        seeds = [123456789, 987654321]
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            runner.secrets, "randbits", side_effect=seeds
        ) as entropy, mock.patch.object(
            runner, "FullHorizonParameterMCTS", wraps=runner.FullHorizonParameterMCTS
        ) as search:
            generated = [runner.main(arguments) for _ in seeds]

        self.assertEqual(entropy.call_args_list, [mock.call(128), mock.call(128)])
        self.assertEqual([call.kwargs["seed"] for call in search.call_args_list], seeds)
        self.assertEqual([result["config"]["mcts_seed"] for result in generated], seeds)
        for result in generated:
            self.assertEqual(result["config"]["seed"], 7)
            self.assertEqual(result["config"]["match_seed"], 123)
        self.assertEqual(
            generated[0]["training"]["ddcfr"]["history"],
            generated[1]["training"]["ddcfr"]["history"],
        )
        actions = [
            [tuple(row[key] for key in ("alpha", "beta", "gamma", "tau"))
             for row in result["training"]["mcts"]["history"]]
            for result in generated
        ]
        self.assertNotEqual(actions[0], actions[1])

        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            runner.secrets, "randbits"
        ) as entropy:
            reproduced = [
                runner.main(arguments + ["--mcts-seed", str(seed)]) for seed in seeds
            ]

        entropy.assert_not_called()
        for expected, actual in zip(generated, reproduced):
            for agent in ("ddcfr", "mcts"):
                self.assertEqual(
                    expected["training"][agent]["history"],
                    actual["training"][agent]["history"],
                )
            self.assertEqual(expected["matches"], actual["matches"])
            self.assertEqual(expected["exact"], actual["exact"])

    def test_default_hand_count_is_balanced_and_cli_reports_totals_and_metrics(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = runner.main([
                "--baseline", "fixed",
                "--iterations", "1",
                "--simulations", "1",
                "--max-depth", "1",
                "--tau", "1",
            ])

        hands = result["config"]["hands"]
        self.assertEqual(result["matches"]["hands"], hands)
        self.assertEqual(result["matches"]["by_seat"]["first"]["games"], hands // 2)
        self.assertEqual(result["matches"]["by_seat"]["second"]["games"], hands // 2)
        self.assertEqual(re.findall(r"(?m)^第\d+戦\s*\|", output.getvalue()), [])
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 14)
        self.assertTrue(output.getvalue().startswith(f"対戦結果（{hands}戦）\n"))
        self.assertEqual(lines[6], "")
        self.assertEqual(lines[7], "厳密な対戦期待値（チップ/戦、先手・後手を半分ずつ）")
        self.assertEqual(lines[10], "")
        self.assertEqual(lines[11], "Exploitability（小さいほど戦略の隙が小さい）")
        for line, label, expected in (
            (lines[8], "DDCFR", result["exact"]["ddcfr_mean_return"]),
            (lines[9], "DDCFR+MCTS", result["exact"]["mcts_mean_return"]),
            (lines[12], "DDCFR", result["training"]["ddcfr"]["exploitability"]),
            (lines[13], "DDCFR+MCTS", result["training"]["mcts"]["exploitability"]),
        ):
            actual_label, actual_value = line.split(":", 1)
            self.assertEqual(actual_label.strip(), label)
            self.assertAlmostEqual(float(actual_value), expected, places=6)
        self.assertNotIn("学習", output.getvalue())
        for absent in ("損益", "95%", "CI", "mean_return", "exploitability="):
            self.assertNotIn(absent, output.getvalue())

    def test_cli_rejects_odd_hands_before_training(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                runner.main(["--baseline", "fixed", "--hands", "99"])
        self.assertEqual(caught.exception.code, 2)

    def test_bundled_learned_model_runs_to_requested_iteration(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.main(
                [
                    "--baseline", "learned",
                    "--iterations", "3",
                    "--simulations", "2",
                    "--max-depth", "1",
                    "--tau", "2",
                    "--hands", "10",
                    "--seed", "3",
                ]
            )

        for agent in ("ddcfr", "mcts"):
            training = result["training"][agent]
            self.assertEqual(training["iterations"], 3)
            self.assertEqual(training["history"][-1]["end_iteration"], 3)
            self.assertTrue(math.isfinite(training["exploitability"]))
            self.assertGreater(training["exploitability"], 0.0)
        self.assertEqual(result["matches"]["hands"], 10)


if __name__ == "__main__":
    unittest.main()
