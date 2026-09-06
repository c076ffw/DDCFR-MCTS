import math
import random

from ddcfr.mcts.simulator import simulate_action


class MCTSNode:
    """
    MCTSの1ノード。

    1つのノード = ある時点のDDCFR solver状態
    1つのedge = (alpha, beta, gamma, tau)
    """

    def __init__(
        self,
        solver,
        actions,
        parent=None,
        action=None,
        depth=0,
    ):
        self.solver = solver

        self.parent = parent
        self.action = action
        self.depth = depth

        self.children = []

        # まだこのノードから試していないparameter
        self.untried_actions = list(actions)

        # MCTS統計量
        self.visits = 0
        self.value_sum = 0.0

    @property
    def mean_value(self):
        if self.visits == 0:
            return 0.0

        return self.value_sum / self.visits


class ParameterMCTS:
    """
    DDCFRのdiscount parametersを選択するMCTS。
    """

    def __init__(
        self,
        actions,
        simulations=50,
        max_depth=3,
        exploration_constant=math.sqrt(2),
        seed=0,
    ):
        self.actions = list(actions)

        self.simulations = simulations
        self.max_depth = max_depth

        self.exploration_constant = exploration_constant

        self.random = random.Random(seed)

        if self.simulations < len(self.actions):
            raise ValueError(
                "simulations should be at least "
                "the number of actions."
            )

    # ==================================================
    # MCTS本体
    # ==================================================

    def search(self, solver):

        # 本物solverの現在exploitability
        root_conv = solver.calc_conv()

        root = MCTSNode(
            solver=solver,
            actions=self.actions,
            depth=0,
        )

        for _ in range(self.simulations):

            node = root

            # ==========================================
            # 1. Selection
            # ==========================================

            while (
                node.depth < self.max_depth
                and len(node.untried_actions) == 0
                and len(node.children) > 0
            ):
                node = self._select_child(node)

            # ==========================================
            # 2. Expansion
            # ==========================================

            if (
                node.depth < self.max_depth
                and len(node.untried_actions) > 0
            ):
                node = self._expand(node)

            # ==========================================
            # 3. Rollout / Evaluation
            # ==========================================

            final_conv = self._rollout(node)

            reward = self._calculate_reward(
                root_conv,
                final_conv,
            )

            # ==========================================
            # 4. Backpropagation
            # ==========================================

            self._backpropagate(
                node,
                reward,
            )

        # ==============================================
        # 最終action決定
        # ==============================================

        best_child = max(
            root.children,
            key=lambda child: (
                child.visits,
                child.mean_value,
            ),
        )

        stats = []

        for child in sorted(
            root.children,
            key=lambda child: child.visits,
            reverse=True,
        ):
            stats.append(
                {
                    "action": child.action,
                    "visits": child.visits,
                    "mean_value": child.mean_value,
                }
            )

        return best_child.action, stats

    # ==================================================
    # Selection
    # ==================================================

    def _select_child(self, node):

        best_child = None
        best_score = -float("inf")

        for child in node.children:

            if child.visits == 0:
                score = float("inf")

            else:
                exploitation = child.mean_value

                exploration = (
                    self.exploration_constant
                    * math.sqrt(
                        math.log(node.visits)
                        / child.visits
                    )
                )

                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    # ==================================================
    # Expansion
    # ==================================================

    def _expand(self, node):

        action = self.random.choice(
            node.untried_actions
        )

        node.untried_actions.remove(action)

        # そのparameterを仮想的に実行
        next_solver, _, _ = simulate_action(
            node.solver,
            action,
        )

        child = MCTSNode(
            solver=next_solver,
            actions=self.actions,
            parent=node,
            action=action,
            depth=node.depth + 1,
        )

        node.children.append(child)

        return child

    # ==================================================
    # Rollout
    # ==================================================

    def _rollout(self, node):

        current_solver = node.solver
        current_depth = node.depth

        # max_depthまでparameterをランダム選択
        while current_depth < self.max_depth:

            action = self.random.choice(
                self.actions
            )

            current_solver, _, _ = simulate_action(
                current_solver,
                action,
            )

            current_depth += 1

        return current_solver.calc_conv()

    # ==================================================
    # Reward
    # ==================================================

    def _calculate_reward(
        self,
        before_conv,
        after_conv,
    ):

        eps = 1e-12

        # exploitabilityの対数改善量
        #
        # afterが小さいほどrewardが大きい
        return (
            math.log10(before_conv + eps)
            - math.log10(after_conv + eps)
        )

    # ==================================================
    # Backpropagation
    # ==================================================

    def _backpropagate(
        self,
        node,
        reward,
    ):

        while node is not None:

            node.visits += 1
            node.value_sum += reward

            node = node.parent