import copy
from numbers import Integral

from ddcfr.cfr import DDCFRSolver


def clone_solver(solver):
    """
    現在のDDCFR solverの学習状態をコピーした
    simulation用solverを作る。
    """

    # 新しいsolverを作成
    sim_solver = DDCFRSolver(
        solver.game_config,
        solver.logger,
    )

    # CFRの学習状態をdeep copy
    sim_solver.states = copy.deepcopy(solver.states)

    # iterationなどの状態もコピー
    sim_solver.num_iterations = solver.num_iterations
    sim_solver.num_nodes_touched = solver.num_nodes_touched
    sim_solver.conv_history = list(solver.conv_history)

    # 存在する場合のみコピー
    if hasattr(solver, "last_num_nodes_touched"):
        sim_solver.last_num_nodes_touched = solver.last_num_nodes_touched

    return sim_solver


def apply_action(solver, action):
    """指定したparameterをtau iterationだけsolverへ適用する。

    solverを直接更新する。探索では必ずコピーしたsolverに適用する。
    exploitabilityは計算せず、呼び出し側で必要な時点に評価する。
    """

    alpha, beta, gamma, tau = action
    if isinstance(tau, bool) or not isinstance(tau, Integral) or tau <= 0:
        raise ValueError("tau must be a positive integer.")

    for _ in range(tau):
        solver.num_iterations += 1
        solver.iteration(alpha=alpha, beta=beta, gamma=gamma)


def simulate_action(solver, action):
    """
    solverを壊さずに、指定した
    (alpha, beta, gamma, tau)
    を仮想的に実行する。
    """

    # 本物とは別のsolverを作る
    sim_solver = clone_solver(solver)

    # 実行前のexploitability
    conv_before = sim_solver.calc_conv()

    # tau iterationだけ仮想実行
    apply_action(sim_solver, action)

    # 実行後のexploitability
    conv_after = sim_solver.calc_conv()

    return sim_solver, conv_before, conv_after
