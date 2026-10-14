import copy
import math
import unittest
from unittest import mock

from ddcfr.mcts.full_horizon_mcts import FullHorizonParameterMCTS
from ddcfr.mcts.simulator import apply_action


class FakeSolver:
    """Deterministic learning state with a configurable exploitability curve."""

    def __init__(self, evaluator=None, num_iterations=0):
        self.num_iterations = num_iterations
        self.num_nodes_touched = 2 * num_iterations
        self.last_num_nodes_touched = self.num_nodes_touched
        self.conv_history = [1.0]
        self.states = {"info": {"regret": 0.0}}
        self.actions = []
        self.on_iteration = None
        self.evaluator = evaluator or (
            lambda solver: 1.0 / (solver.num_iterations + 1)
        )

    def iteration(self, alpha, beta, gamma):
        self.actions.append((alpha, beta, gamma, self.num_iterations))
        self.states["info"]["regret"] += alpha + beta + gamma
        self.num_nodes_touched += 2
        if self.on_iteration is not None:
            self.on_iteration(self.num_iterations)

    def calc_conv(self):
        return self.evaluator(self)


def learning_state(solver):
    snapshot = {
        "num_iterations": solver.num_iterations,
        "num_nodes_touched": solver.num_nodes_touched,
        "conv_history": copy.deepcopy(solver.conv_history),
        "states": {
            key: copy.deepcopy(
                state if isinstance(state, dict) else state.__dict__
            )
            for key, state in solver.states.items()
        },
    }
    if hasattr(solver, "last_num_nodes_touched"):
        snapshot["last_num_nodes_touched"] = solver.last_num_nodes_touched
    if hasattr(solver, "actions"):
        snapshot["actions"] = list(solver.actions)
    return snapshot


class FullHorizonParameterMCTSTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            "ddcfr.mcts.full_horizon_mcts.clone_solver",
            side_effect=copy.deepcopy,
        )
        self.clone = patcher.start()
        self.addCleanup(patcher.stop)

    def test_rollouts_reach_exact_horizon_beyond_tree_depth(self):
        evaluated_iterations = []

        def evaluate(solver):
            evaluated_iterations.append(solver.num_iterations)
            return 1.0 / (solver.num_iterations + 1)

        solver = FakeSolver(evaluate, num_iterations=2)
        simulated_iterations = []
        solver.on_iteration = lambda iteration: simulated_iterations.append(iteration)
        before = learning_state(solver)
        mcts = FullHorizonParameterMCTS(
            total_iterations=10,
            simulations=5,
            max_depth=1,
            pw_k=1.0,
            pw_alpha=0.0,
            tau=3,
            seed=7,
        )

        action, stats = mcts.search(solver)

        self.assertEqual(action[3], 3)
        self.assertEqual(stats["start_iteration"], 2)
        self.assertEqual(stats["total_iterations"], 10)
        self.assertEqual(stats["simulations"], 5)
        self.assertAlmostEqual(stats["initial_conv"], 1.0 / 3)
        self.assertAlmostEqual(stats["best_conv"], 1.0 / 11)
        self.assertEqual(evaluated_iterations, [2] + [10] * 5)
        self.assertEqual(stats["simulated_iterations"], len(simulated_iterations))
        # The root edge is cached; only its five-iteration suffix is replayed.
        self.assertEqual(stats["simulated_iterations"], 3 + 5 * 5)
        self.assertEqual(learning_state(solver), before)

    def test_selects_root_action_by_terminal_not_short_term_conv(self):
        # Alpha 1 wins after one iteration; alpha 2 wins after two iterations.
        def evaluate(solver):
            if not solver.actions:
                return 1.0
            first_alpha = solver.actions[0][0]
            if solver.num_iterations == 1:
                return 0.1 if first_alpha == 1.0 else 0.8
            return 0.5 if first_alpha == 1.0 else 0.05

        solver = FakeSolver(evaluate)
        mcts = FullHorizonParameterMCTS(
            total_iterations=2,
            simulations=2,
            max_depth=1,
            pw_k=2.0,
            tau=1,
        )
        action_a = (1.0, 0.0, 0.0, 1)
        action_b = (2.0, 0.0, 0.0, 1)
        with mock.patch.object(
            mcts,
            "_sample_action",
            side_effect=[action_a, action_a, action_b, action_b],
        ):
            action, stats = mcts.search(solver)

        self.assertEqual(action, action_b)
        self.assertAlmostEqual(stats["best_conv"], 0.05)
        self.assertAlmostEqual(stats["best_reward"], math.log10(20), places=8)
        self.assertEqual(solver.num_iterations, 0)
        self.assertEqual(solver.actions, [])

    def test_robust_root_choice_can_differ_from_best_observed_rollout(self):
        def evaluate(solver):
            if not solver.actions:
                return 1.0
            if solver.actions[0][0] == 2.0:
                return 0.1
            return 0.01 if solver.actions[-1][0] == 0.0 else 1.0

        mcts = FullHorizonParameterMCTS(
            total_iterations=2,
            simulations=7,
            max_depth=1,
            pw_k=2.0,
            pw_alpha=0.0,
            tau=1,
        )
        action_a = (1.0, 0.0, 0.0, 1)
        action_b = (2.0, 0.0, 0.0, 1)
        lucky_suffix = (0.0, 0.0, 0.0, 1)
        usual_suffix = (3.0, 0.0, 0.0, 1)
        actions = [action_a, lucky_suffix, action_b] + [usual_suffix] * 6
        with mock.patch.object(mcts, "_sample_action", side_effect=actions):
            action, stats = mcts.search(FakeSolver(evaluate))

        self.assertEqual(action, action_b)
        self.assertAlmostEqual(stats["best_conv"], 0.01)
        self.assertEqual(stats["root_stats"][0]["action"], action_b)
        self.assertEqual(stats["root_stats"][0]["visits"], 4)

    def test_last_action_is_clipped_to_remaining_iterations(self):
        solver = FakeSolver(num_iterations=6)
        action, stats = FullHorizonParameterMCTS(
            total_iterations=7,
            simulations=1,
            max_depth=1,
            tau=2,
        ).search(solver)

        self.assertEqual(action[3], 1)
        self.assertEqual(stats["simulations"], 1)
        self.assertEqual(stats["simulated_iterations"], 1)
        self.assertAlmostEqual(stats["best_conv"], 1.0 / 8)

    def test_same_seed_reproduces_action_and_statistics(self):
        options = dict(
            total_iterations=11,
            simulations=8,
            max_depth=2,
            tau=3,
            seed=42,
        )
        first = FullHorizonParameterMCTS(**options).search(FakeSolver())
        second = FullHorizonParameterMCTS(**options).search(FakeSolver())

        self.assertEqual(first, second)

    def test_finished_solver_has_no_next_action(self):
        solver = FakeSolver(num_iterations=5)
        before = learning_state(solver)
        mcts = FullHorizonParameterMCTS(total_iterations=5, simulations=3)

        with self.assertRaises(ValueError):
            mcts.search(solver)

        self.assertEqual(learning_state(solver), before)
        self.clone.assert_not_called()

    def test_horizon_before_current_iteration_is_rejected(self):
        solver = FakeSolver(num_iterations=6)
        before = learning_state(solver)

        with self.assertRaises(ValueError):
            FullHorizonParameterMCTS(total_iterations=5).search(solver)

        self.assertEqual(learning_state(solver), before)
        self.clone.assert_not_called()

    def test_simulation_budget_counts_actual_iterations_without_short_rollouts(self):
        evaluated_iterations = []
        simulated_iterations = []

        def evaluate(solver):
            evaluated_iterations.append(solver.num_iterations)
            return 1.0 / (solver.num_iterations + 1)

        solver = FakeSolver(evaluate, num_iterations=2)
        solver.on_iteration = lambda iteration: simulated_iterations.append(iteration)
        action, stats = FullHorizonParameterMCTS(
            total_iterations=13,
            simulations=20,
            max_depth=1,
            tau=4,
            simulation_budget=22,
        ).search(solver)

        self.assertEqual(action[3], 4)
        self.assertGreaterEqual(stats["simulations"], 1)
        self.assertLess(stats["simulations"], 20)
        self.assertEqual(stats["simulated_iterations"], len(simulated_iterations))
        self.assertLessEqual(stats["simulated_iterations"], 22)
        self.assertEqual(
            evaluated_iterations,
            [2] + [13] * stats["simulations"],
        )
        self.assertEqual(solver.num_iterations, 2)

    def test_budget_too_small_for_complete_rollout_is_rejected(self):
        solver = FakeSolver(num_iterations=2)

        with self.assertRaises(ValueError):
            FullHorizonParameterMCTS(
                total_iterations=13,
                simulation_budget=10,
            ).search(solver)

        self.clone.assert_not_called()

    def test_apply_action_advances_supplied_solver_without_evaluation(self):
        solver = FakeSolver(num_iterations=2)
        solver.calc_conv = mock.Mock(side_effect=AssertionError("unexpected eval"))

        apply_action(solver, (1.5, -1.0, 2.0, 3))

        self.assertEqual(solver.num_iterations, 5)
        self.assertEqual(
            solver.actions,
            [(1.5, -1.0, 2.0, iteration) for iteration in (3, 4, 5)],
        )
        self.assertEqual(solver.num_nodes_touched, 10)
        self.assertEqual(solver.conv_history, [1.0])
        solver.calc_conv.assert_not_called()


class FullHorizonParameterMCTSIntegrationTest(unittest.TestCase):
    def test_kuhn_full_horizon_search_preserves_source(self):
        from ddcfr.cfr import DDCFRSolver
        from ddcfr.game.game_config import KuhnPoker
        from ddcfr.utils.logger import Logger

        logger = Logger(writer_strings=[])
        self.addCleanup(logger.close)
        solver = DDCFRSolver(KuhnPoker(iterations=11), logger)
        apply_action(solver, (1.5, 0.0, 2.0, 2))
        before = learning_state(solver)
        action, stats = FullHorizonParameterMCTS(
            total_iterations=11,
            simulations=6,
            max_depth=1,
            tau=4,
            seed=2,
        ).search(solver)

        self.assertEqual(learning_state(solver), before)
        self.assertEqual(action[3], 4)
        self.assertEqual(stats["start_iteration"], 2)
        self.assertEqual(stats["total_iterations"], 11)
        self.assertEqual(stats["simulations"], 6)
        self.assertTrue(math.isfinite(stats["best_conv"]))
        self.assertGreater(stats["best_conv"], 0)

        apply_action(solver, action)
        self.assertEqual(solver.num_iterations, 6)

    def test_runner_replans_after_each_real_block(self):
        from scripts import run_continuous_mcts_parameter_search as runner

        original_search = FullHorizonParameterMCTS.search
        search_start_iterations = []

        def search_and_record(mcts, solver):
            search_start_iterations.append(solver.num_iterations)
            return original_search(mcts, solver)

        with mock.patch.object(
            FullHorizonParameterMCTS,
            "search",
            autospec=True,
            side_effect=search_and_record,
        ) as search, mock.patch("builtins.print"):
            result = runner.main(
                [
                    "--iterations", "9",
                    "--simulations", "3",
                    "--max-depth", "1",
                    "--tau", "4",
                    "--seed", "3",
                ]
            )

        self.assertEqual(search.call_count, 3)
        self.assertEqual(search_start_iterations, [0, 4, 8])
        self.assertEqual(result["iterations"], 9)
        self.assertEqual(len(result["selected_history"]), 3)
        self.assertTrue(math.isfinite(result["final_conv"]))


if __name__ == "__main__":
    unittest.main()
