# バルブ36条件の保存範囲

正式269版は31/36成功（86.1%）。同じ入力・乱数で構造生成から実行した最終版だけを集計し、計画失敗2件も分母に含める。詳細は `final36_v269/audit_summary.json` と `engineering_review.json`。

## Gitへ保存したもの

- 固定入力、ソースのハッシュ、環境、全36件の集計、失敗・接触役割・離脱後動作の確認、構造画像。
- `validation269.tar.xz`：正式試験の元記録。各条件の生成計画、物理モデル、初期・最終状態、途中の物理状態、指令発行・入力計算の実測時刻、実行時ソースの写しを含む。
- `validation269_manifest.json`：圧縮ファイルと全収録ファイルのSHA-256。元記録は変更せず保存した。
- 追加監査用の4スクリプト、最終的なProレビュー67・69・70、コミット前の159検査の記録。

途中版の大量の実験出力はGitへ追加していない。ソースの写しには実行時に記録対象だった生成済み形状・補助ファイルも含む。これは当時の入力の証拠であり、現在の実装へ採用するファイル一覧ではない。

## 別のチェックアウトで元記録を復元する

この保存モデルは `/home/leus/amsrr_det` の絶対パスを含む。同じ配置先で、リポジトリ直下から実行する。

```bash
tar --skip-old-files -xJf artifacts/valve_diversity_20261031/validation269.tar.xz
mkdir -p build
cp -an artifacts/valve_diversity_20261031/final36_v269/source_snapshot/build/collision_physx_docks build/
```

既存ファイルは上書きしない。形状は保存されたものを使う。保存ソースや共有ライブラリを稼働中のソースへ一括コピーする必要はない。計算環境は `environment.json`、物理エンジンの変更は `third_party/mujoco/manifest.json` と `contact.patch` を参照する。集計器は実際にロードされた物理ライブラリのSHA-256も確認し、異なる場合は不整合を報告する。同じライブラリ版名だけで同一の物理実装と扱わない。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m scripts.report_valve_panel artifacts/valve_diversity_20261031/final36_v269
```

これは保存された状態・指令時刻の再確認であり、実験の再実行ではない。新規の36条件実行はREADMEのコマンドを使用する。計算機や実測遅延が変われば成功数も同一とは限らない。

## 判定の範囲

- 成功台数は1・2・4・5・7・8。3・6機はこの最終試験で未実行。
- 非鉛直は6Nm以下、高負荷48/60Nmは鉛直のみ。高負荷の成功は9/12。
- 指令発行は最大1.735ms、10ms超過0。観測から新結果作用は最大31.875msであり、その間も物理を進めた。
- 開発で再利用した固定入力での確認。未知条件、VIM4、実機、変更後の把持150条件を検証した結果ではない。
