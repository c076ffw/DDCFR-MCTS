import math
import random

from ddcfr.mcts.simulator import simulate_action


class ContinuousMCTSNode:
    """
    Continuous MCTS の1ノード。

    node:
        ある時点のDDCFR solver状態

    edge:
        (alpha, beta, gamma, tau)
    """

    def __init__(
        self,
        solver,
        parent=None,
        action=None,
        depth=0,
    ):
        self.solver = solver

        self.parent = parent
        self.action = action
        self.depth = depth

        self.children = []

        # MCTS統計
        self.visits = 0
        self.value_sum = 0.0

    @property
    def mean_value(self):

        if self.visits == 0:
            return 0.0

        return self.value_sum / self.visits


class ContinuousParameterMCTS:
    """
    DDCFRの(alpha, beta, gamma)を
    連続空間から探索するMCTS。

    Progressive Wideningによって、
    nodeのvisit数に応じて新しいparameterを追加する。
    """

    def __init__(
        self,
        simulations=100,
        max_depth=3,
        exploration_constant=math.sqrt(2),
        pw_k=2.0,
        pw_alpha=0.5,
        alpha_range=(0.0, 5.0),
        beta_range=(-5.0, 0.0),
        gamma_range=(0.0, 5.0),
        tau=5,
        seed=0,
    ):

        self.simulations = simulations
        self.max_depth = max_depth

        # UCB
        self.exploration_constant = exploration_constant

        # Progressive Widening
        self.pw_k = pw_k
        self.pw_alpha = pw_alpha

        # parameter探索範囲
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range

        # 最初はtau固定
        self.tau = tau

        self.random = random.Random(seed)

    # ==================================================
    # MCTS本体
    # ==================================================

    def search(self, solver):

        root_conv = solver.calc_conv()

        root = ContinuousMCTSNode(
            solver=solver,
            depth=0,
        )

        for _ in range(self.simulations):

            node = root

            # ==========================================
            # Selection + Progressive Widening
            # ==========================================

            while node.depth < self.max_depth:

                # 新しいactionを追加できるならExpansion
                if self._can_expand(node):

                    node = self._expand(node)
                    break

                # 追加できないなら既存childをUCBで選ぶ
                if len(node.children) > 0:

                    node = self._select_child(node)

                else:
                    # 念のため
                    node = self._expand(node)
                    break

            # ==========================================
            # Rollout
            # ==========================================

            final_conv = self._rollout(node)

            # ==========================================
            # Evaluation
            # ==========================================

            reward = self._calculate_reward(
                root_conv,
                final_conv,
            )

            # ==========================================
            # Backpropagation
            # ==========================================

            self._backpropagate(
                node,
                reward,
            )

        # ==================================================
        # rootのchildから最終parameterを決める
        # ==================================================

        if len(root.children) == 0:
            raise RuntimeError(
                "MCTS did not generate any actions."
            )

        # Robust child:
        # 最も多く探索されたparameterを採用
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
    # Progressive Widening
    # ==================================================

    def _can_expand(self, node):
        """
        |children(s)| < k * N(s)^alpha

        を満たしていれば、
        新しいparameterを生成してよい。
        """

        max_children = self._max_children(node)

        return len(node.children) < max_children

    def _max_children(self, node):

        # visits=0でも最低1個は生成可能にする
        allowed = (
            self.pw_k
            * ((node.visits + 1) ** self.pw_alpha)
        )

        return max(
            1,
            int(math.ceil(allowed)),
        )

    # ==================================================
    # Continuous parameter sampling
    # ==================================================

    def _sample_action(self):

        alpha = self.random.uniform(
            self.alpha_range[0],
            self.alpha_range[1],
        )

        beta = self.random.uniform(
            self.beta_range[0],
            self.beta_range[1],
        )

        gamma = self.random.uniform(
            self.gamma_range[0],
            self.gamma_range[1],
        )

        return (
            alpha,
            beta,
            gamma,
            self.tau,
        )

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
                        math.log(
                            max(1, node.visits)
                        )
                        / child.visits
                    )
                )

                score = (
                    exploitation
                    + exploration
                )

            if score > best_score:

                best_score = score
                best_child = child

        return best_child

    # ==================================================
    # Expansion
    # ==================================================

    def _expand(self, node):

        # 連続空間から新しいparameter生成
        action = self._sample_action()

        # 仮想的にDDCFRを進める
        next_solver, _, _ = simulate_action(
            node.solver,
            action,
        )

        child = ContinuousMCTSNode(
            solver=next_solver,
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

        while current_depth < self.max_depth:

            # Rolloutでも連続parameterを生成
            action = self._sample_action()

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