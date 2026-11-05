# 歩行36条件の最終記録

正式評価は33/36成功（91.7%）、成功例平均ホバー代理比39.52%、全初期計画最大14.459秒。固定3〜8機・22/26/30cm/s・直進と左右カーブ・5mの同じ36条件を、新規構造生成から実行した。制御計算の実測遅延中も旧指令で物理を進めている。

## 保存内容

- `final36_attitude/` 直下の入力、環境、ソース識別、集計、失敗診断、最終確認。
- `halfstep36_attitude/step_comparison.json`：同じ保存36計画を物理刻み0.25msで再実行した比較。元の0.5msと全合否が一致し、成功例平均39.49%。新規生成や初期計画時間の証拠へ合算しない。
- `validation_final.tar.xz`：上記2評価の各条件の計画、物理モデル、初期・最終・途中状態、実測計算時刻、当時のソースの写し。途中版の評価は含めない。
- `validation_manifest.json`：圧縮ファイルと全2,466収録ファイルのサイズ・SHA-256。圧縮後に全収録内容を照合済み。
- 二足3機、三足4機、四足6/8機の4動画と周期画像。表示用の床格子とカメラ以外は保存状態の再描画で、物理や制御の再試行ではない。
- `engineering_review.py`、`review_halfstep.py`、`render_final_walks.py`：保存結果の確認・描画手順。`run_halfstep.py` は刻み確認時の実行入口。
- `final_audit_contract.json`、`pro_round176.txt`、実験前102検査とコミット前320検査の記録。

ソースの写しには、その時点で作業場所にあった補助ファイルや生成済み形状も含む。実験時の来歴を確認する記録であり、現行の採用実装一覧ではない。コミット整理では2ファイルの行末空白だけを除去し、Pythonの構文木が不変であることを確認した。旧・新SHA-256は `commit_cleanup.json` に記録し、実験時ソースの写しは変更していない。

## 復元と確認

保存モデルは `/home/leus/amsrr_det` の絶対パスを含む。同じ配置先のリポジトリ直下で、既存ファイルを上書きせず復元する。

```bash
tar --skip-old-files -xJf artifacts/walking_speed8_20261104/validation_final.tar.xz
mkdir -p build
cp -an artifacts/walking_speed8_20261104/final36_attitude/source_snapshot/build/collision_physx_docks build/
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m scripts.report_walking_panel artifacts/walking_speed8_20261104/final36_attitude
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python artifacts/walking_speed8_20261104/engineering_review.py artifacts/walking_speed8_20261104/final36_attitude
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python artifacts/walking_speed8_20261104/review_halfstep.py artifacts/walking_speed8_20261104/final36_attitude artifacts/walking_speed8_20261104/halfstep36_attitude
```

物理ライブラリの環境は `final36_attitude/environment.json` と `third_party/mujoco/manifest.json` を参照。保存ソースや共有ライブラリを現行ソースへ一括コピーする必要はない。上のコマンドは保存結果の集計であり、物理実験を再実行しない。

4動画を再描画する場合は、EGLとffmpegを利用できる環境で次を実行する。既存の同名動画・周期画像を上書きする。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python artifacts/walking_speed8_20261104/render_final_walks.py artifacts/walking_speed8_20261104/final36_attitude
```

新規生成からの再実験はREADMEのコマンドを使う。PCや実測遅延が変われば、結果の一致は保証しない。

## 結果の範囲

- 足数2〜8の初期適合候補を生成し、今回の最終採用は2〜4足。5〜8足の実歩行成功は未確認。
- 3機高速2件の支持時間不足、8機高速直進1件の足以外の荷重超過は失敗として残す。集中検定の非検出を、弱さの不存在の証明としない。
- 指令発行最大3.838ms・10ms超過0。新観測の作用遅延は別に最大13msで物理へ反映済み。PCでの測定であり、実機・VIM4・OSの厳密な時間保証ではない。
- ホバー代理比は未校正の推力指標。実電力量や他機体の論文を上回った証拠ではない。
