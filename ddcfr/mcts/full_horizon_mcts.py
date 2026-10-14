"""DDCFRのparameterを、最終iterationの収束度を見て逐次選択する。"""

import math
import random
from numbers import Integral

from ddcfr.mcts.simulator import apply_action, clone_solver


class FullHorizonMCTSNode:
    """solverの学習状態と、そこへ至るaction・探索統計を保持する。"""

    def __init__(self, solver, parent=None, action=None, depth=0):
        self.solver = solver
        self.parent = parent
        self.action = action
        self.depth = depth
        self.children = []
        self.visits = 0
        self.value_sum = 0.0

    @property
    def mean_value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class FullHorizonParameterMCTS:
    """最終iterationまで仮実行し、次の区間のparameterを決める。

    max_depthは探索木の深さだけを制限する。各rolloutは必ず最終iteration
    まで実行し、そのexploitabilityで評価する。実行後のsolverから再探索
    するため、返すのはrootのactionだけ。

    simulation_budgetは一回の探索で仮実行するDDCFR iteration数の上限。
    評価期間を短縮せず、完全なrolloutの回数を抑えて予算内に収める。
    """

    def __init__(
        self,
        total_iterations,
        simulations=100,
        max_depth=50,
        exploration_constant=math.sqrt(2),
        pw_k=2.0,
        pw_alpha=0.5,
        alpha_range=(0.0, 5.0),
        beta_range=(-5.0, 0.0),
        gamma_range=(0.0, 5.0),
        tau=5,
        seed=0,
        simulation_budget=None,
    ):
        for name, value, minimum in (
            ("total_iterations", total_iterations, 0),
            ("simulations", simulations, 1),
            ("max_depth", max_depth, 1),
            ("tau", tau, 1),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if simulation_budget is not None and (
            isinstance(simulation_budget, bool)
            or not isinstance(simulation_budget, Integral)
            or simulation_budget <= 0
        ):
            raise ValueError("simulation_budget must be a positive integer or None.")
        if not math.isfinite(pw_k) or pw_k <= 0:
            raise ValueError("pw_k must be finite and positive.")
        if not math.isfinite(pw_alpha) or not 0 <= pw_alpha <= 1:
            raise ValueError("pw_alpha must be between 0 and 1.")
        if not math.isfinite(exploration_constant) or exploration_constant < 0:
            raise ValueError("exploration_constant must be finite and nonnegative.")
        for name, bounds in (
            ("alpha_range", alpha_range),
            ("beta_range", beta_range),
            ("gamma_range", gamma_range),
        ):
            if (
                len(bounds) != 2
                or not all(math.isfinite(value) for value in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"{name} must contain finite, ordered bounds.")

        self.simulations = simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.pw_k = pw_k
        self.pw_alpha = pw_alpha
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        self.tau = tau
        self.random = random.Random(seed)
        self.total_iterations = total_iterations
        self.simulation_budget = simulation_budget

    def search(self, solver):
        """Return (next_action, stats) without advancing the real solver.

        total_iterations is an absolute target, including any iterations already
        completed by solver. The final action is shortened to fit this target.
        best_conv reports the best terminal rollout observed during this search;
        future searches can choose a different continuation.
        """
        remaining = self.total_iterations - solver.num_iterations
        if remaining <= 0:
            raise ValueError("total_iterations must be after the solver's current iteration.")

        simulation_limit = self.simulations
        if self.simulation_budget is not None:
            # 木に保存した区間の再利用で実際の消費量はこれより小さくなる。
            simulation_limit = min(simulation_limit, self.simulation_budget // remaining)
            if simulation_limit == 0:
                raise ValueError(
                    "simulation_budget must allow at least one full rollout "
                    f"({remaining} remaining iterations)."
                )

        root_conv = solver.calc_conv()
        root = FullHorizonMCTSNode(solver=solver)
        best_conv = float("inf")
        self._simulated_iterations = 0

        for _ in range(simulation_limit):
            node = root
            while (
                node.solver.num_iterations < self.total_iterations
                and node.depth < self.max_depth
            ):
                if self._can_expand(node):
                    node = self._expand(node)
                    break
                node = self._select_child(node)

            final_conv = self._rollout(node)
            reward = self._calculate_reward(root_conv, final_conv)
            self._backpropagate(node, reward)
            best_conv = min(best_conv, final_conv)

        best_child = max(
            root.children,
            key=lambda child: (child.visits, child.mean_value),
        )
        stats = {
            "initial_conv": root_conv,
            "best_conv": best_conv,
            "best_reward": self._calculate_reward(root_conv, best_conv),
            "simulations": root.visits,
            "simulated_iterations": self._simulated_iterations,
            "start_iteration": solver.num_iterations,
            "total_iterations": self.total_iterations,
            "root_stats": [
                {
                    "action": child.action,
                    "visits": child.visits,
                    "mean_value": child.mean_value,
                }
                for child in sorted(
                    root.children,
                    key=lambda child: (child.visits, child.mean_value),
                    reverse=True,
                )
            ],
        }
        return best_child.action, stats

    def _can_expand(self, node):
        return len(node.children) < self._max_children(node)

    def _max_children(self, node):
        # 訪問回数に応じて候補を増やし、未訪問でも最低1個は生成する。
        allowed = self.pw_k * ((node.visits + 1) ** self.pw_alpha)
        return max(1, int(math.ceil(allowed)))

    def _sample_action(self):
        alpha = self.random.uniform(self.alpha_range[0], self.alpha_range[1])
        beta = self.random.uniform(self.beta_range[0], self.beta_range[1])
        gamma = self.random.uniform(self.gamma_range[0], self.gamma_range[1])
        return alpha, beta, gamma, self.tau

    def _select_child(self, node):
        best_child = None
        best_score = -float("inf")
        for child in node.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploration = self.exploration_constant * math.sqrt(
                    math.log(max(1, node.visits)) / child.visits
                )
                score = child.mean_value + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _sample_remaining_action(self, solver):
        alpha, beta, gamma, tau = self._sample_action()
        remaining = self.total_iterations - solver.num_iterations
        return alpha, beta, gamma, min(tau, remaining)

    def _expand(self, node):
        action = self._sample_remaining_action(node.solver)
        next_solver = clone_solver(node.solver)
        apply_action(next_solver, action)
        self._simulated_iterations += action[3]
        child = FullHorizonMCTSNode(
            solver=next_solver,
            parent=node,
            action=action,
            depth=node.depth + 1,
        )
        node.children.append(child)
        return child

    def _rollout(self, node):
        current_solver = node.solver
        if current_solver.num_iterations < self.total_iterations:
            # rollout全体で一度だけ複製し、木ノードの学習状態を保つ。
            current_solver = clone_solver(current_solver)
            while current_solver.num_iterations < self.total_iterations:
                action = self._sample_remaining_action(current_solver)
                apply_action(current_solver, action)
                self._simulated_iterations += action[3]

        return current_solver.calc_conv()

    def _calculate_reward(self, before_conv, after_conv):
        eps = 1e-12
        return math.log10(before_conv + eps) - math.log10(after_conv + eps)

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent
