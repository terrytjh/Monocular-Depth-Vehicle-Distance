# 共同盲測事前登錄(方法 B):comma2k19 第 8 包整包 208 段,第九版(v9d 為主要結果、v9 另報)

登錄人:terrytjh(徐鐽睿)。本文件與共用 repo 的 `methods/B_depth_lane_ruler/PREREG_comma2k19_chunk8.md` 內容相同。

## 0. 背景

- 共用 repo `monocular-driving-distance/Monocular-Measurement-of-Driving-Distance-Based-on-AI-Depth-Estimation-Models`:
  方法 A(chennnlee)與方法 B(本 repo)在同一批資料、同一真值、同一評分器上比較。規則見該 repo 的 `docs/SPEC.md` 與
  `docs/DECISIONS.md` D1–D12(2026-10-18 全部定案,`255db91`)。本文件只登錄方法 B。
- 第 8 包在共用 repo 的分割表(`splits/comma2k19.json`)標為 blind,兩邊都沒有用過。

## 1. 方法(預測程式一行都不改)

- 程式:共用 repo 的 `methods/B_depth_lane_ruler/`:本 repo `5aff285` 的複本(109 個檔逐一相同;`tools/` 與第九版登錄 `586cfeb` 相同)
  加上 `run.py`、`dev_constants.py`(`526f77c`)。`run.py` 在工作資料夾照本 repo 的配置跑 `tools/run_v9.sh`,只改三支腳本開頭的 `cd` 一行,
  第九版自己的封存步驟跳過,改用共用 repo 的封存。
- **v9d(主要結果)**:第九版,只把相機 k 改成只從開發包學(`dev_constants.py`;規則同 `tools/camera_k.py learn` 與 `learn-night`:
  同一台相機、排除同一趟、至少 5 段)。開發包 = 本 repo 的 `blind_v4`、`blind_v5`、`blind_v7`、`blind_v8`、`blind_v9`
  (Chunk_4、5、6、7 前半、7 後半);同為開發資料的 Chunk_3(`blind_v2`)沒有放進來,2026-10-18 實作與檢查 v9d 時就是這五包。
  相機 A 照第九版(`blind_v3`–`blind_v8`)。對應共用 repo D10:主要結果的跨段常數只能從開發包學。
- **v9(另報)**:第九版登錄原樣,相機 k 從本批自己的預測學(不看真值)。
- 兩版共用同一份追蹤與深度快取(DA3、MoGe-2),v9d 只重做用到相機 k 的步驟。
- 輸入:共用格式的 `segment.json`(路類與日夜 = `splits/batches/comma2k19_chunk8_labels.csv`,`e0d6077`)與影格
  (imageio-ffmpeg 0.6.0 附的 ffmpeg 7.0.2 static,`-f hevc -fps_mode passthrough -qscale:v 2`)。
- 輸出規則同第九版(量到 60 m 以外的讀值不給數字)。轉成共同格式照 `run.py`:車道號 +1 = 左;自車速的負值記 0 並標 `clipped_to_0`。

## 2. 資料

- comma2k19 第 8 包整包:208 段、19 趟,全是第二台車(Hugging Face `commaai/comma2k19` `raw_data/Chunk_8.zip`,
  9,634,728,776 bytes,官方 sha256 `3b7dd599681895815c8666838df580d37ae408a72bdc1fabe72e3e276ec8c323`)。
  批次清單 `splits/batches/comma2k19_chunk8.txt`(`e0d6077`)。
- 組別依判定表:**主要組 = 白天 `ca_highway` + `unknown`,126 段**;白天一般道路(`ca_local`)43 段;夜間 39 段。
- 已知的暴露(都在本文件之前):
  - 影格已解出,只讀 `video.hevc` 與 `global_pose/frame_times`(核對張數)。
  - 路類與日夜由看圖者判定:每段只看第 10、30、50 秒三張影格,兩輪獨立判讀,不一致的 13 段判第三輪取多數。
    判讀之外,只有在確認版面時開過第一張對照表。
  - 段名與日期(2018-08-08 到 2018-10-06)。
  - 沒有任何預測,沒有任何真值。
- 全部跑、全部評分,只跑一次。

## 3. 評分

- **主要**:共用 repo 的評分器與定案的設定(`configs/eval_comma2k19.json`,封存時以雜湊記錄)。
  - 真值:第二台車 = 雷達處理後距離 − 0.70 m(後輪接地點)。
  - 配對:中心欄 615(第二台車相機重裝後);封存前兩邊各量每趟的消失點,某趟兩邊都偏離 615 超過 10 px 時改用兩邊的平均;
    582 當敏感度,同一次評分重新配對。
  - 其他照 D1–D12:去掉空槽、合併重複回報、車寬閘 0.8–4.5 m、不用距離閘、移動 ≥ 7.2 km/h(定位車速)、依真值距離分段、
    自車速每秒(覆蓋率含拒發段)、相對速度用該幀雷達值、整段重抽的 95% 區間。
- **次要**(只拿來和本 repo 第七到九版對照,不當主要結果):封存後再用本 repo 的 `tools/v9_score.py`,照第九版登錄的規則
  (雷達 − 0.33 m、50% 閘)評一次。

### 3.1 事前寫下的預期(不論有沒有達到都照實報告)

依據:我自己照定案的共同規則重寫的評分(不是共用 repo 的程式;`/mnt/d/depth_work/joint/chunk8/prereg_dev/`),
在已開過的第 7 包上算得:v9 後半 4.12%(0.83 m)、v9d 後半 3.95%(0.80 m)、v9 前半(登錄前試跑)3.83%(0.76 m)。
本 repo 自己的規則下(雷達 − 0.33 m),各包之間相差到 0.8 個百分點(第七到九版 4.21–4.98%)。

**主要組(126 段),v9d,主要設定:**

1. 真值 30 m 內中位誤差 ≤ 4.8%,且 ≤ 0.95 m;中位偏差在 ±3% 內。
2. 10–20 m 中位誤差 ≤ 0.75 m、20–30 m ≤ 1.15 m(開發 0.57–0.69 m、0.92–0.98 m)。
3. 有輸出的段 ≥ 主要組的 90%(開發 54 / 54)。
4. 相對速度(真值 30 m 內)MAE ≤ 2.3 km/h(開發 1.76–2.03);95% 範圍的實際覆蓋 90–99%。
5. 自車速每秒 MAE ≤ 3.5 km/h(開發 2.94、3.47);每幀 95% 範圍的實際覆蓋 90–99%;秒數覆蓋率(含拒發段)照報,不設門檻。
6. v9d 與 v9 在同一批讀值上的中位誤差相差 ≤ 0.3 個百分點(兩者只差相機 k,開發上約 1%;方向不預設)。
7. 配對中心 615 與 582 相比:找到率較高,誤差 > 25% 的比例較低(開發三組都是)。

**白天一般道路(43 段):**

8. v9d 中位誤差 ≤ 5.0%(開發 3.09%、4.74%)。

**夜間(39 段):**

9. v9d 中位誤差 ≤ 7.0%(開發 3.83%、6.50%,讀值少、各包差異大)。

## 4. 流程

1. 本文件 commit 在本 repo 與共用 repo(內容相同),Terry 推上兩邊的 GitHub。
2. 影格與消失點(都不碰真值,也不改任何預測):
   - 預測開始前,各段影格的 `frames_sha256` 已照共用規則算好;兩邊 `segment.json` 的合併雜湊(本機
     `c2656b4c7f22120dcee2c4d97d8407b2ad0a340003cde6654a8457d592af6db2`)在封存前核對一致,不一致先查原因,影格不同就重跑預測。
   - 每趟的消失點取自本方法預測時自己的車道線結果(`*_run.json` 的交點),在封存前交給評分設定。
3. 預測:`run.py --segments-root <segments> --ids splits/batches/comma2k19_chunk8.txt --out-root predictions/B_depth_lane_ruler/v9`,
   再加 `--camera-k-from-dev` 產生 v9d(同一份快取)。不讀任何答案。
4. 封存:照共用 repo `docs/SPEC.md` 第 6 節,每個版本一份封存檔,封存後立刻推上。
5. 兩個方法都封存之後才產生真值;評分;結果寫進 `docs/JOINT_CHUNK8_RESULTS.md`,第 3.1 節逐條照實。

## 5. 開發時看過答案的資料(不能當證據)

第九版登錄第 5 節列的全部,加上第九版盲測的 Chunk_7 後半 101 段(2026-10-16 評分後開過)。第 8 包沒有。
