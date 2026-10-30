# 新制御の検証記録

固定150条件を実測遅延込みで実行した168版は131件成功（87.3%）。計算待ちの間も旧指令で物理時間を進めた。全周期10ms以内は未達で、613269周期中1回、18.224msの発行超過がある。

## Gitに保存するもの

- `final150_v168/audit_summary.json`: 独立再確認の結果、各条件の成否、条件別の集計、時間と偏りの検査。不整合0件。
- `final150_v168/run_summary.json`: 全150条件の実行一覧と実行中のソース変更0件の記録。
- 同ディレクトリの `failure_diagnosis.json`、`comparison_v103.json`、`environment.json`、`evaluation_scope.json`、`source_manifest.json`、`auditor_manifest.json`、`panel.json`: 失敗分類、改善前比較、計算環境、評価範囲、実験時のソース識別、条件一覧。
- `replay_inputs168.tar.gz` と `replay_inputs168_manifest.json`: 再実行に必要な148件の生成構造・生成時の閉姿勢・物理形状と固定150条件の一覧。元の計画欠損2件は復元せず、失敗として数える。各入力のバイト列のSHA-256とアーカイブのSHA-256を保存する。
- `collision_adoption103.json`: 接触計算方式の独立検査と採用根拠。
- `pro_question37.txt` と `pro_round37.png`: 共通制御の適用範囲に関するProとの最終確認。
- `commit_cleanup_tests.txt`: コミット整理後の128件の関連テスト結果。

詳細な全周期の遅延ログ・実験時の全ソースコピー・途中試作はローカルの実験ディレクトリに残すが、今回のGit追加には含めない。保存した集計だけから全周期の監査を再計算できるという意味ではない。新しい実行の記録は次の手順で作成し、監査できる。

## 再実行

リポジトリを `/home/leus/amsrr_det` に配置し、既存の依存環境構築を行う。保存シーンはこの絶対パスを含む。新しいチェックアウトで入力がない場合のみ、リポジトリ直下で展開する。

```bash
tar --keep-old-files -xzf artifacts/relative_control_20261028/replay_inputs168.tar.gz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/run_relative_panel.py artifacts/new_relative150 --jobs 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/report_relative_panel.py artifacts/new_relative150
```

実行先は存在しないディレクトリを指定する。CPU番号は既定で発行側0/4/8/12、計算側2/6/10/14。異なるCPU構成では `--cores` と `--worker-cores` を変更し、時間と成功率をその環境の結果として扱う。実測遅延を使うため、同じ入力でも結果が必ず同一になる保証はない。

監査は実行時のソースと現在のソースを照合する。実行完了後にコードを変更する前に行い、古い記録を現行ソースで無条件に再監査しない。

## コミット整理の範囲

却下済み制御向けの新しい実行処理と代替動作クラスは除外した。新制御も使っていた最終成功判定は、条件を変えず `amsrr_det/task_audit.py` へ移した。関数本体の構文比較で同一を確認し、関連テスト128件が成功した。168版の150件を整理後に再実行したわけではない。旧コマンド `build/check_local_task.py` は以前の比較用実行器のまま残し、現方式の入口には `build/check_relative_task.py` を使う。
