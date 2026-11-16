# DDCFR 対 DDCFR＋MCTS

`play_ddcfr_vs_mcts.py` は、Kuhn Poker で両者を独立に学習させ、学習後の平均戦略を使って対戦させる追加スクリプトです。各戦を終局まで進め、最後に通算対戦結果と、両者の対戦期待値・exploitability を表示します。先手・後手を半分ずつ担当し、100 戦なら `--hands 100` を指定します。既存の実装・モデル・結果ファイルは変更しません。対戦中は戦略を更新しません。

- DDCFR：付属の学習済み ES モデル `models/model.pkl` が割引パラメータと適用期間を決めます。
- DDCFR＋MCTS：`FullHorizonParameterMCTS` が割引パラメータを選択し、各ブロックを適用した後に再探索します。各 rollout は指定の最終イテレーションまで進めます。

MCTS は学習時の割引パラメータ探索に使います。対戦中のベット選択には、両者とも学習後の平均戦略を使います。

## 実行

リポジトリのルートで実行します。既存の DDCFR 環境を使います。

```bash
conda activate DDCFR
python scripts/play_ddcfr_vs_mcts.py
```

短い動作確認：

```bash
python scripts/play_ddcfr_vs_mcts.py \
  --iterations 20 --simulations 5 --max-depth 2 --tau 5 \
  --hands 100 --seed 0
```

標準設定は両者それぞれ 1000 イテレーション、対戦は合計 10000 戦です。`--hands` に 2 以上の偶数を指定すると対戦数を変更できます。MCTS は最大 12 シミュレーション、木の深さ 5、ブロック長 20。探索範囲は実際の既存スクリプト `run_continuous_mcts_parameter_search.py` と同じ `alpha=[1,2]`, `beta=[-1,0]`, `gamma=[1,3]` です。

MCTS の探索と、対戦時のカード配布・行動選択には、それぞれ実行ごとに新しい乱数シードを使います。`--seed` は DDCFR 側の学習用で、既定値は `0` です。MCTS の探索シードは `--mcts-seed`、対戦のシードは `--match-seed` で個別に指定できます。省略したものはそれぞれ自動生成します。通算勝敗が偶然同じになることはあります。

学習から対戦まで再現したい場合は、同じ学習設定に加えて、MCTS と対戦の両方のシードを指定します。

```bash
python scripts/play_ddcfr_vs_mcts.py --hands 100 --seed 0 --mcts-seed 7 --match-seed 42
```

結果は実行ごとに `results/DDCFR_vs_MCTS/<UTC日時>/` に自動保存します。保存先を変更する場合は、**まだ存在しないディレクトリ**を指定します。

```bash
python scripts/play_ddcfr_vs_mcts.py \
  --iterations 100 --hands 100 --output results/duel_001
```

- `result.json`：設定 (`config`)、両者の学習履歴・時間・exploitability (`training`)、対戦の勝敗集計 (`matches`)、厳密な対戦期待値 (`exact`)。
- `matches.csv`：対戦番号 (`hand`)、DDCFR の席 (`ddcfr_seat`、0 が先手・1 が後手)、勝者 (`winner`)。
- `training.csv`：両者が各ブロックで使用したパラメータ、適用区間、exploitability、MCTS の探索量。
- `convergence.csv`：両者の初期状態と各学習ブロック終了時の `agent, iteration, exploitability`。
- `convergence.png`：DDCFR と DDCFR＋MCTS の収束曲線を重ねたグラフ。横軸は実際の学習イテレーション数、縦軸は exploitability の対数目盛で、小さいほど戦略の隙が小さいことを表します。

収束曲線には実際に学習した solver の平均戦略の評価値を使い、MCTS 内の仮実行の値は含めません。学習済みモデルでは選択されるブロック長が変わるため、両者の評価時点が異なる場合があります。各点の間は線で結びます。

実際に使用した MCTS の探索シードと対戦シードは、`result.json` の `config.mcts_seed` と `config.match_seed` に記録します。自動生成されたシードも `--mcts-seed` と `--match-seed` に指定すれば再利用できます。

`--output` を省略すると実行ごとに新しい保存先を作ります。指定した保存先が既に存在する場合は、学習開始前にエラーにして上書きを防ぎます。

## 結果の読み方

各戦でカードを新たに配り、両者の平均戦略から行動を抽選し、フォールドまたはショーダウンで決着するまで進めます。先手・後手を 1 戦ごとに交代します。

全対戦の終了後に、両者の通算の勝ち・負け・引き分け、DDCFR が先手のとき・後手のときそれぞれの勝敗、両者の対戦期待値と exploitability を表示します。画面には学習進捗や各試合の勝敗を表示しません。

通算対戦結果の表示例（続けて両者の期待値と exploitability を表示します）：

```text
対戦結果（100戦）
DDCFR      : 46勝 54敗
DDCFR+MCTS : 54勝 46敗
引き分け   : 0戦
DDCFRが先手（50戦）: 24勝 26敗 0分
DDCFRが後手（50戦）: 22勝 28敗 0分
```

各試合の勝者も `matches.csv` に保存します。

対戦期待値は、今回学習した同じ二つの平均戦略について、全カード配布・全行動を確率で重み付けして計算します。先手・後手を半分ずつ担当したときの 1 戦あたりの期待利益で、単位はチップ、正が利益です。抽選した対戦の実測平均や勝率ではありません。ゼロ和ゲームなので両者の期待値の和は、浮動小数点の誤差を除いてゼロになります。`exact.ddcfr_as_player0` と `exact.ddcfr_as_player1` には DDCFR の席別期待値も保存します。

Exploitability は、相手をこの二者に限定しない戦略の隙の大きさで、小さいほどナッシュ均衡に近いことを示します。各 solver の平均戦略全体を評価した値で、今回の相手への期待利益や勝率とは別の指標です。学習の最終ブロックで評価した値を表示・保存します。既存 solver に合わせて下限は `1e-12` です。

同じ学習イテレーション数で比較しますが、MCTS には追加の探索計算があります。両者の学習時間には solver の生成などの準備を含み、学習済み DDCFR ではモデルの読み込みも含みます。

## 設定変更

```bash
# 既存の固定パラメータ版 (alpha=1.5, beta=0, gamma=2) と比較
python scripts/play_ddcfr_vs_mcts.py --baseline fixed --iterations 100

# 別の学習済み ES モデルを使用
python scripts/play_ddcfr_vs_mcts.py --model path/to/model.pkl

# 探索範囲を拡大
python scripts/play_ddcfr_vs_mcts.py \
  --alpha-range 0 5 --beta-range -5 0 --gamma-range 0 5
```

`--simulation-budget` は一回の探索で仮実行する DDCFR イテレーション数の予算です。既定は `max(iterations, simulations * max_depth * tau)`。省略しない場合は `--iterations` 以上を指定し、`0` で制限を解除します。終端を短縮せず、完了するシミュレーション数を調整します。探索は有限回の近似です。

```bash
python scripts/play_ddcfr_vs_mcts.py --help
python -m unittest tests.test_ddcfr_vs_mcts
```
