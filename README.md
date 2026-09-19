# DDCFR-MCTS

> Dynamic Discounted Counterfactual Regret Minimization with Monte Carlo Tree Search <br>
> DDCFRにおける割引パラメータの決定方法をMCTSに変更した実装

## Overview

本リポジトリでは、Dynamic Discounted Counterfactual Regret Minimization (DDCFR) をベースに、**割引パラメータの決定方法をMonte Carlo Tree Search (MCTS) に変更した手法**を実装・検証する。

元のDDCFRでは、学習したモデルを用いて割引スキームを決定する。本研究では、この決定方法をMCTSによる探索に変更する。

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
