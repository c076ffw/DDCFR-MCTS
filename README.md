# DDCFR-MCTS

> Dynamic Discounted Counterfactual Regret Minimization with Monte Carlo Tree Search <br>
> DDCFRにおける割引パラメータの決定方法をMCTSに変更した実装

## Overview

本リポジトリでは、Dynamic Discounted Counterfactual Regret Minimization (DDCFR) をベースに、**割引パラメータの決定方法をMonte Carlo Tree Search (MCTS) に変更した手法**を実装・検証する。

元のDDCFRでは、学習したモデルを用いて割引スキームを決定する。本研究では、この決定方法をMCTSによる探索に変更する。

## 全イテレーションを考慮したMCTSによるパラメータ探索

Kuhn Pokerで、MCTSにより次のブロックの `(alpha, beta, gamma, tau)` を選び、`tau` イテレーション進めてから再探索する。各探索では、現在のイテレーションから指定した**最終イテレーションまで**solverのコピーを進め、終端のexploitabilityを評価する。これにより、ブロックごとの逐次選択を続けながら、最終結果を考慮してパラメータを決める。

```bash
conda activate DDCFR
python scripts/run_continuous_mcts_parameter_search.py
```

デフォルトは1000イテレーション、最大100シミュレーション、木の深さ50、`tau=5`、乱数seed 0。パラメータの探索範囲は `alpha: [0, 5]`、`beta: [-5, 0]`、`gamma: [0, 5]`。`tau` イテレーションの間は同じパラメータを使い、最後のブロックは残りのイテレーション数に合わせて短くする。

小さい設定で動作確認する場合は、コマンドラインから設定を変更できる。

```bash
python scripts/run_continuous_mcts_parameter_search.py \
  --iterations 20 --simulations 5 --max-depth 2 --tau 5 --seed 0
```

報酬は、探索開始時と終端のexploitabilityの常用対数の差。`--max-depth` はMCTSの木を保持する深さの上限で、rolloutの終了地点ではない。木の深さ上限に達しても、rolloutは `--iterations` で指定した終端まで続ける。最後に、根の子のうち訪問回数が最も多いパラメータを次のブロックに採用する。

探索ごとの計算量は `--simulation-budget` で制限できる。これは探索中に進めるDDCFRイテレーション数の予算で、未指定時は `max(iterations, simulations * max_depth * tau)`（デフォルト設定では25000）。残りイテレーション数を `R`、予算を `B` とすると、シミュレーション数を `min(simulations, B // R)` に減らす。評価する終端は短くしない。予算は少なくとも1シミュレーションを完了できる値、すなわち初回は `--iterations` 以上にする必要がある。`--simulation-budget 0` を指定すると予算による制限を解除する。

予算なしの場合、全体イテレーション数 `N`、シミュレーション数 `S`、従来の探索先読み幅 `H = max_depth * tau` とすると、各ブロックで探索を繰り返す従来方式の概算は `S * N * H / tau`、終端まで評価する方式は `S * N * (N + tau) / (2 * tau)`（`N` が `tau` の倍数の場合）。デフォルト設定では、それぞれ約500万と1005万の仮想イテレーションに相当する。これは木の共有やsolverの複製・評価コストを含まない概算であり、実行時間を保証するものではない。既定の予算は、従来の1回の探索の名目上の計算量に合わせてシミュレーション数を調整する。

選択するパラメータは有限回の探索による近似であり、連続パラメータ空間の大域的な最適解は保証しない。各ブロックで選んだパラメータ、実際に完了したシミュレーション数・仮想イテレーション数、最良rolloutの終端exploitabilityを表示する。最良rolloutの値は探索中に評価した続きを使った診断値であり、その後の再探索を含む実行全体の最終値の予測ではない。実行後には実測の最終exploitabilityと各ブロックの履歴を表示する。

実行設定とブロックごとの再探索は `scripts/run_continuous_mcts_parameter_search.py`、探索木・候補生成・UCB選択・終端評価・報酬の集計は `ddcfr/mcts/full_horizon_mcts.py`、solverの複製とパラメータ適用は `ddcfr/mcts/simulator.py` が担当する。旧連続MCTSの実装と専用確認スクリプトは削除し、必要な共通処理を `full_horizon_mcts.py` に集約している。

最終時点までの仮実行、元のsolverの状態保持、予算制限、ブロックごとの再探索は、次のテストで確認できる。

```bash
python -m unittest tests.test_full_horizon_mcts
```

## Original DDCFR

本リポジトリは、以下のDDCFR実装をベースとしている。

[Original DDCFR Repository]https://github.com/rpSebastian/DDCFR.git

> Dynamic Discounted Counterfactual Regret Minimization <br>
> Hang Xu<sup>*</sup> , Kai Li<sup>*,#</sup>, Haobo Fu, Qiang Fu, Junliang Xing<sup>#</sup>, Jian Cheng <br>
> ICLR 2024 (Spotlight)

## Branches

* `baseline-original`: 元のDDCFR実装

---

# Dynamic Discounted Counterfactual Regret Minimization

> Dynamic Discounted Counterfactual Regret Minimization <br>
> Hang Xu<sup>*</sup> , Kai Li<sup>*,#</sup>, Haobo Fu, Qiang Fu, Junliang Xing<sup>#</sup>, Jian Cheng <br>
> ICLR 2024 (Spotlight)

## Install DDCFR

Install miniconda3 from [the official website](https://docs.conda.io/en/latest/miniconda.html) and run the following script:

```shell
bash scripts/install.sh
```

## Train DDCFR

We use games implemented by [OpenSpiel](https://github.com/deepmind/open_spiel) [1] and [PokerRL](https://github.com/EricSteinberger/PokerRL) [2]. To easily run the code for training, we provide a unified interface. Each experiment will generate an experiment id `logid` and create a unique directory in `logs/es`. Models will be stored in the folder `logs/es/{log_id}/model`. Run the following script to start training.

```bash
conda activate DDCFR
python scripts/train_es.py with save_log=True
```


# Dynamic Discounted Counterfactual Regret Minimization

> Dynamic Discounted Counterfactual Regret Minimization <br>
> Hang Xu<sup>\*</sup> , Kai Li<sup>\*,\#</sup>, Haobo Fu, Qiang Fu, Junliang Xing<sup>#</sup>, Jian Cheng <br>
> ICLR 2024 (Spotlight)

## Install DDCFR

Install miniconda3 from [the official website](https://docs.conda.io/en/latest/miniconda.html) and run the following script:

```shell
bash scripts/install.sh
```

## Train DDCFR

We use games implemented by [OpenSpiel](https://github.com/deepmind/open_spiel) [1] and [PokerRL](https://github.com/EricSteinberger/PokerRL) [2].  To easily run the code for training, we provide a unified interface. Each experiment will generate an experiment id `logid` and create a unique directory in `logs/es`.  Models will be stored in the folder `logs/es/{log_id}/model `.  Run the following script to start training.
```bash
conda activate DDCFR
python scripts/train_es.py with save_log=True
```

You can modify the configuration in the following ways:

1. Modify the Markov decision process. Edit the file `ddcfr/cfr/cfr_env.py`.
2. Modify the type and number of training games. Specify your game in `ddcfr/game/game_config.py:get_train_configs`.
3. Modify the hyperparameters. Edit the file `scripts/train_es.py`.
4. Train on distributed servers. Follow the instruction of [ray](https://docs.ray.io/en/master/cluster/cloud.html#cluster-private-setup) to set up your private cluster and set `ray.init(address="auto")` in `scripts/train_es.py`.
5. Train DDCFR using Reinforcement Learning instead of Evolutionary Strategies. Run the following script:
```bash
conda activate DDCFR
python scripts/run_ppo.py with save_log=True
```


## Test learned discounting schemes by DDCFR

Run the following script to test discounting schemes learned by DDCFR, where `logid` is the generated unique experiment id, and `model_id` is the selected model id. The results are saved in the folder `results`.

```bash
bash scripts/test_es_model.sh {experiment id} {model_id}
```

If the model is trained using Reinforcement Learning, run the following script instead:
```bash
bash scripts/test_rl_model.sh {experiment id} {model_id}
```

## Test the learned discounting scheme in paper

Run the following script to test learned discounting scheme in paper. The results are saved in the folder `results`.
```bash
bash scripts/test_learned_model.sh
```

## Citing
If you use DDCFR in your research, you can cite it as follows:
```
@inproceedings{DDCFR,
  title     = {Dynamic Discounted Counterfactual Regret Minimization},
  author    = {Hang, Xu and Kai, Li and Haobo, Fu and Qiang, Fu and Junliang, Xing and Jian Cheng},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  pages     = {1--18}
}
```

## References

[1] Lanctot, M.; Lockhart, E.; Lespiau, J.-B.; Zambaldi, V.; Upadhyay, S.; P´erolat, J.; Srinivasan, S.; Timbers, F.; Tuyls, K.; Omidshafiei, S.; Hennes, D.; Morrill, D.; Muller, P.; Ewalds, T.; Faulkner, R.; Kram´ar, J.; Vylder, B. D.; Saeta, B.; Bradbury, J.; Ding, D.; Borgeaud, S.; Lai, M.; Schrittwieser, J.; Anthony, T.; Hughes, E.; Danihelka, I.; and Ryan-Davis, J. 2019. OpenSpiel: A Framework for Reinforcement Learning in Games. CoRR, abs/1908.09453.

[2] Steinberger, E. 2019. PokerRL. https://github.com/TinkeringCode/PokerRL.
