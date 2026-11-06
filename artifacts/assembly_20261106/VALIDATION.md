# 空中組立の評価

**採用版 `full219_central27` は全219件終了、時間条件込み218件成功（99.5%）。** 実接続210/211、単機8/8。全219件で物理組立と最終保持は完了した。1件の離陸時に発行処理12.527ms、観測間隔20msとなったため、その件は失敗として残す。全周期10ms以内・厳密な実時間保証は未達。開発版・旧独立制御・保存状態の診断成功は合算しない。

組立の計画準備max5.99884秒、初回ホバー確認後の完了max22.94秒。発行357012回中10ms超1回、成功218例の発行max4.05349ms。新観測の作用遅延max23msを物理へ反映した。台数別成功は1〜8機順に8/8、8/8、93/93、42/42、24/24、19/20、14/14、10/10。11事前比較は全Holm p=1、非検出を等率の証明としない。

全件の物理状態監査・手順監査は不整合0、並行組立98件。関連43検査成功。半刻み6件は6/6成功・監査不整合0、2N/3N横力の別2件も各1回の退避・再接近から15.44/22.20秒で成功・監査不整合0。気流対照は下記のとおり主成績と分離する。

[全件の判定](full219_central27/audit.json) / [時間と切替](full219_central27/engineering_review.json) / [手順監査](full219_central27/schedule_audit.json) / [刻み感度](halfstep6_audit_central27/audit.json) / [外乱回復](gust_audit_central27/audit.json) / [ソース・入力の一致](final_integrity27.json)。1603実行関連ファイルと219入力の現行/保存SHA-256が一致。

## 固定入力

`configs/evaluation/assembly_generated.json` に、元ファイルのSHA-256、接続木、初期配置乱数を保存した。

- 把持149：`artifacts/runs/control_tail150_v8_20261027/*/generated_plan/plan.json`
- バルブ35：`artifacts/valve_side_faces_20261101/full36_v15/*/selection.json`
- 歩行35：`artifacts/walking_landing_20261105/final36_v13/*/selection.json`

元タスク実行で失敗した例も、構造が生成できていれば含む。1機8例は離陸・保持の対照、接続を行うのは211例。各機数の件数は1〜8機の順に8、8、93、42、24、20、14、10。

## 判定

初期は地面で停止し、xy/yawを無作為に決める。各機の最寄り機までの水平中心距離1〜2m。同型機の役割割当は変更できるが、初期位置・向きの集合は変えない。

全機のホバリング確認から、接続・退避・再接近・制御切替・最後の保持まで30秒以内。捕捉は位置・角度・相対速度条件を100ms観測してから接続口の拘束を有効にし、物理位置・速度をリセットしない。最終保持は制御切替完了後、固定した世界座標で全機の位置4cm・角度3度・速度0.06m/s・角速度0.1rad/s以内を1秒、接続口のずれも検査する。

計算中も旧指令で物理を進める。100Hzの指令発行処理と、新観測から新指令作用までの遅延は分ける。発行10ms超過は当該ケースの失敗へ含める。組立計画時間は保存済みの確定構造から手順・制御準備を作る時間であり、元タスクの新規構造生成との合算15秒を確認した評価ではない。

## 実行と独立確認

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m scripts.run_assembly_panel artifacts/my_assembly_panel
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m scripts.report_assembly_panel artifacts/my_assembly_panel
```

実行はCPU組0/2、4/6、8/10の3並列。各ケースを一度だけ実行し、180秒の外部上限を設定する。ソース・環境・入力を保存して途中変更を検査する。

独立集計は、保存された物理状態から接続条件・初期位置/yaw・保持・遅延・切替初回の出力一致・古い構造の計算結果が指令へ入らないことを確認する。台数別・元タスク別の11比較に片側Fisher検定とHolm補正を用いる。非検出は等しい失敗率や未知条件への保証ではない。

追加の手順監査は、記録した有限の経路族について、合法な同時接続集合の欠落がないこと、同一部分構造の作業重複がないこと、選択した公称時間がそのグラフの最小値であることを照合する。あらゆる連続経路・実時間の最短性は保証しない。

## 別集計の感度・外乱・映像

主成績に加算せず、失敗の差し替えにも使わない。

- 事前指定6例を、同じ初期条件・CPU組で物理刻み0.5ms→0.25msとして順次実行する。
- 3機の最初の接近中に0.35秒の横力2N、3Nを与える。両方CPU4/6で実行し、追加外乱からの減速・退避・再接近を検査する。
- 別の2機5秒保持で、上下配置・気流なし、上下配置・気流あり、水平1m離隔・気流ありを比較する。
- 保存した物理状態から動画を描画し、接続順と各部分構造のリーダー切替を確認する。動画生成のために物理を再実行しない。

上下2機の最終保持誤差は、気流なし0.130mm、気流あり0.4087m（保持不成立）、横1m離隔・気流あり0.153mm。通常組立での推力低下max7.42%、上下対照ではmax62.0%。任意の強後流に耐える結果とは扱わない。

気流モデルは未校正で、地面効果や詳細な渦・機体遮蔽を含まない。PC上のリーダー/指令所有を模擬しており、実Wi-Fi通信・実機のラッチ・機体強度・実時間OSは未検証。[方式](../../for_codex/ASSEMBLY_METHOD.md)を参照。

Pro192〜194で要求統一・中央配分・切替と評価範囲を確認した。最終確認は[Pro194](pro_round194.txt)。切替中の予定Dock作用と実出力の差、発行時間超過1件、PC上の検証範囲を残した報告であり、全条件成功や全周期10ms保証へ読み替えない。今回の実験は終了。

## 保存と復元

最終219件・半刻み6件・横力2件・気流対照3件の元記録、実行時ソース、および入力に使った確定構造を `validation_01.tar.xz`〜`validation_09.tar.xz` に保存した。`validation_manifest.json` に全4,815ファイルのSHA-256を記録し、圧縮後に全内容の一致を確認した。動画と集計結果は直接参照できる形でも保存している。途中の試作・棄却案の実験結果は追加していない。実行時ソースの複製は来歴を記録するもので、含まれる全補助ファイルを採用実装とする意味ではない。

保存モデルは `/home/leus/amsrr_det` のパスを含む。同じ場所でリポジトリを復元し、既存ファイルを上書きせず展開する。

```bash
for archive in artifacts/assembly_20261106/validation_*.tar.xz; do
  tar --skip-old-files -xJf "$archive"
done
mkdir -p build
cp -an artifacts/assembly_20261106/full219_central27/source_snapshot/build/collision_physx_docks build/
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python artifacts/assembly_20261106/audit_physics.py artifacts/assembly_20261106/full219_central27
```

追加監査の参照リンクは `audit_sensitivity.py` と `audit_gusts.py` で再作成できる。これらは保存記録の監査用で、物理実験を起動しない。`render_central.py` は保存状態から動画を生成する。依存環境は `full219_central27/environment.json` を参照。コミット前には組立・共通飛行・ローター配分の関連44検査が成功した（`commit_tests.log`）。
