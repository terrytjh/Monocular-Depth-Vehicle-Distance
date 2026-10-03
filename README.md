# Monocular-Depth-Vehicle-Distance

**Monocular Vehicle Distance Measurement Based on AI Depth Estimation Models**

**以 AI 深度估算模型為基礎的單鏡頭影像行車距離量測方法**

從一台行車紀錄器的影片，量出**前車與左右車道車輛的距離**、**這些車的相對速度與絕對速度**，以及**自車速**。
每一個速度都附 95% 範圍。用到的只有影片本身、公開的深度模型與偵測器，以及道路上的法定標線。

**研究歷程**:國科會大專學生研究計畫(115-2813-C-004-063-E)，計畫書於 2026 年寒假撰寫。2026-09-30 之前的工作屬於團隊專題，
在另一個 repo;本 repo 自 2026-09-30 起只收錄我個人計畫的方法與程式。每一版都先 commit 登錄文件，再預測、封存、評分，
commit 與封存的時間可以逐一核對(`docs/seals/`)。

## 兩個方法

**方法一:深度模型 × 法定虛線尺**(主方法)
單目度量深度模型(Depth Anything 3 metric [3])給每個像素一個距離，但它的公尺尺度會隨相機與場景偏掉。
車道虛線的週期是法規定的(台灣第 182 條 10 m;加州高速公路 Caltrans A20A 14.63 m;美國聯邦 MUTCD 12.19 m)，
所以在**模型自己的深度裡**量出虛線週期 P，就得到尺:`k = 法定週期 ÷ P`，距離 = k × 模型讀值。
自車速是同一條深度剖面上，虛線邊緣在兩張影格之間的平移。

**方法二:標線幾何法，不用深度模型**(對照方法，作法引自文獻 [1, 2],不是本研究的貢獻;程式與早期文件中稱「方法 C」)
平路、針孔相機:畫面第 y 列的路面距離 `d = A / (y − y_h)`。地平線 y_h 取兩條車道線的交點;
A 由虛線在 `u = 1/(y − y_h)` 上的週期換算(`A = 法定週期 ÷ 週期`)。

兩個方法用同一組車輛框與追蹤編號(YOLOv8m-seg + BoT-SORT)，速度、95% 範圍與評分也用同一套程式，
差別只在距離與自車速怎麼來。

## 驗證方式:先登錄、再預測、封存、最後才看答案

1. 方法與參數只在「答案已經開過」的開發片上調整，每次看答案都記在 `docs/TERRY_DEV_LOG.md`。
2. 方法定版後寫成事前登錄(`docs/DEPTH_DASH_V2_PREREG.md`)並 commit;commit 的時間就是登錄時間。
3. 在答案從沒被讀過的盲測片上預測(`tools/run_v2.sh`、`tools/run_v2_av2.sh`)，程式只讀影片。
4. `tools/seal_predictions.py` 把全部預測檔的 sha256 封存，並確認真值檔還不存在。
5. 封存之後才產生真值、評分;評分程式先核對封存雜湊。一批盲測片只用一次。

真值:comma2k19 的原廠雷達、CAN 與定位(`tools/c2k19_extract.py`);Argoverse 2 的光達 3D 框與定位。

## 版本

每一版都先登錄、後盲測;**每一版的結果都保留**，後一版不取代、不刪除前一版的紀錄。Git tag 標出每個里程碑。

| 版本 | 登錄 | 內容 | 盲測資料 | 狀態 |
|---|---|---|---|---|
| 第二版 | 2026-09-30 19:54(`v2-prereg`) | 深度 × 法定虛線尺(量車用近端週期的尺 k_car、k 的保護 1.5、路類由畫面事先判斷決定週期);標線幾何法為對照 | comma2k19 149 段(答案從未讀過)、Argoverse 2 四支 | 預測封存(`v2-sealed`)→ 評分(`v2-results`) |
| 第二版追記 1 | 2026-10-01(`v2-addendum1`) | 方法不變;台灣保留片(人工畫格法)只評自車速 | — | 已評分 |
| 第三版 | 2026-10-01 15:03(`v3-prereg`) | **只改一項**:深度法自車速改用前後 ±2.5 秒的局部尺(含保護)，其餘同第二版 | comma2k19 Chunk_2 194 段(另一台車，答案從未讀過) | 已評分(`docs/DEPTH_DASH_V3_RESULTS.md`) |
| 第四版 | 2026-10-02 04:02(`bec5f39`) | **改用正確的雷達真值**(見下);深度法量車改用路面校正(以標線幾何的路面距離逐段校正比例與偏移)與軌跡平滑 | comma2k19 Chunk_4 205 段、台灣網路行車影片 18 支 | 已評分(`docs/DEPTH_DASH_V4_RESULTS.md`) |
| 第五版 | 2026-10-02 15:17(`ce7d4ec`) | 尺度檢查(擋下兩支尺同時鎖到半週期的段)、兩個方法的自車速平均 | comma2k19 Chunk_5 211 段 | 已評分(`docs/DEPTH_DASH_V5_RESULTS.md`) |
| 第六版 | 2026-10-02 21:46(`819429c`) | **只改一項**:30 m 以外改用每台車在近處量得的「距離 × 框寬」當它自己的尺,與深度距離取幾何平均 | 與第五版同一批(登錄時沒有任何真值) | 已評分(同上) |

文件:各版的 `docs/DEPTH_DASH_V*_PREREG.md`(登錄)與 `docs/DEPTH_DASH_V*_RESULTS.md`(結果);
開發過程(每次看答案的時間與內容)`docs/TERRY_DEV_LOG.md`;封存雜湊 `docs/seals/`。

## 真值的更正(2026-10-02)

comma2k19 處理後的雷達距離已經是 CAN 訊號 LONG_DIST 加上 2.70 m(openpilot 的 RDR_TO_LDR)。用資料集自帶的原始 CAN 逐筆解碼確認
(開發時約 520 萬筆;第四版評分前對 Chunk_4 205 段 3,915,426 筆再確認，全部 2.70 m)，並以法定車牌尺寸與車寬交叉檢查。
第二、三版登錄的距離真值「雷達 + 2.37 m」因此遠了 2.70 m;正確真值 = 雷達 − 0.33 m。**第二、三版的距離數字不能當成準度**
(照登錄保留，事後重算寫在各結果文件文末的追記);自車速、相對速度與他車絕對速度不受影響。第四版起一開始就用正確真值。

## 結果(第五、六版盲測，正確真值，`docs/DEPTH_DASH_V5_RESULTS.md`)

comma2k19 Chunk_5,白天高速公路含看不出來的 125 段(答案從未讀過)。第五版的距離照第四版不變，改的是自車速與絕對速度。
事前寫下的預期:第五版 7 條中 5 條達成，第六版 4 條中 2 條達成。

| | 第五版 | 第四版(同一批) |
|---|---|---|
| 有輸出的段(尺度檢查擋下 7 段) | 94 / 125 | 101 / 125 |
| 自車速每秒 MAE(同一批 4,176 秒) | **3.00 km/h** | 4.50 |
| 他車絕對速度 MAE(同一批 4,793 個時窗) | **3.61 km/h** | 4.38 |
| 他車相對速度 MAE(全部猜 0) | 2.34 km/h(8.47) | 2.35 |
| 本車道 / 左右一道距離，中位誤差 | 6.36% / 5.49%(真值 < 30 m:5.6% / 4.8%) | 6.42% / 5.51% |
| 95% 範圍實際覆蓋(相對 / 絕對 / 自車) | 93 / 98 / 97% | 93 / 99 / 98% |

- 自車速 75 段中 74 段變好(單尾符號檢定 p = 2 × 10⁻²¹)。**未達成**:本車道距離 6.36%(預期 ≤ 6%)、尺度檢查擋下 6.9% 的段(預期 ≤ 6%)。
- **第六版**(30 m 以外、有自己的尺的讀值，佔範圍外的 35%):中位誤差 17.0% → **9.7%**,偏差 −16.6% → −2.9%,86 段中 74 段變好
  (p = 2 × 10⁻¹²);30–40 / 40–50 / 50–60 / 60–80 m 為 5.0 / 5.2 / 9.6 / 18.8%。**未達成**:整體 ≤ 9%、60–80 m ≤ 10%。
- 夜間(42 段)第五版有輸出 23 段，自車速 95% 範圍只覆蓋 74%;第六版在夜間只有 147 筆，變差(14.9 → 22.9%)。

## 結果(第四版盲測，正確真值，`docs/DEPTH_DASH_V4_RESULTS.md`)

comma2k19 Chunk_4,白天高速公路含看不出來的 113 段(答案從未讀過);事前寫下的 10 條預期:8 條達成、1 條部分達成、1 條未達成。

| | 第四版(深度法 + 路面校正) | 第二版(登錄的量車尺) | 標線幾何法 |
|---|---|---|---|
| 有輸出的段 | 84 / 113 | 71 / 113 | 84 / 113 |
| 本車道 / 左右一道距離，中位誤差(偏差) | **5.0%(−3.2%)/ 5.3%(−3.5%)** | 11.5%(+11.0%)/ 12.6%(+12.0%) | 4.7%(−3.5%)/ 5.9%(−5.5%) |
| 同上，只看真值 < 30 m | **4.0% / 4.4%** | 12.1% / 12.9% | 4.3% / 5.5% |
| 他車相對速度 MAE(全部猜 0) | 2.38 km/h(8.67) | 2.27 | 1.78 |
| 他車絕對速度 MAE | **3.90 km/h** | 5.79 | 3.94 |
| 自車速每秒 MAE | **4.55 km/h** | 6.39 | 6.04 |
| 95% 範圍實際覆蓋(相對 / 絕對 / 自車) | 93 / 99 / 98% | 94 / 94 / 93% | 96 / 95 / 92% |

- 同一批雷達配對上，第四版 4.6% 對第二版 12.0%,59 段中 53 段第四版較好(單尾符號檢定 p = 9 × 10⁻¹¹)。
- **第三版**(Chunk_2,82 段):自車速 6.18 → 4.11 km/h(69 段中 67 段變好)、他車絕對速度 5.40 → 3.52 km/h。
- **尺的作用**:同一個深度模型不乘尺時，在焦距與名目值相差大的相機上(Argoverse 2 一支開發片，13 筆配對，初步對照)距離誤差 32%，乘上虛線尺後 5%。
- **限制**:30 m 以外偏短約 19%(只標示、不修正，第六版處理);關掉 50% 配對閘後本車道誤差變大——事後檢查是自己車的引擎蓋被偵測成車(第五版那批去掉後 60.7% → 8.8%,規則還沒登錄);夜間本車道 12%。
- **台灣網路行車影片**(弱真值，只評自車速):18 支只有 2 支畫面上有車速，深度法逐秒差距 3.1 與 9.3 km/h。
  拒發多半是分析範圍下緣固定封頂在畫面高度 0.74,把虛線切掉了(封存後的檢查);新的自動規則(`tools/band_bottom.py`)
  在開發資料上讓標線幾何法有輸出的片從 7 支變 16 支，之後以新蒐集的台灣片驗證。

## 安裝

```bash
python3 -m venv ~/venvs/depthbench && source ~/venvs/depthbench/bin/activate
pip install torch ultralytics opencv-python numpy pandas pyarrow scipy lap openpyxl pillow matplotlib moviepy
```
測試環境:Python 3.13、torch 2.14、ultralytics 8.4.153、opencv-python 4.13。

**Depth models**:官方 repo 放在 `depth_models/`，權重照各 repo 的說明下載:

| 模型 | repo | 測試用的 commit | 授權 |
|---|---|---|---|
| Depth Anything 3(`da3_metric`，主方法) | https://github.com/ByteDance-Seed/Depth-Anything-3 | 3d835ec | Apache 2.0 |
| Metric3D v2 | https://github.com/YvanYin/Metric3D | eb5b6fa | BSD-2(權重授權未載明) |
| UniDepth v2 | https://github.com/lpiccinelli-eth/UniDepth | 8d8cfe4 | CC BY-NC 4.0 |
| Depth Pro | https://github.com/apple/ml-depth-pro | 9e65e4d | Apple AMLR |
| Depth Anything V2(Small) | https://github.com/DepthAnything/Depth-Anything-V2 | a561b84 | Apache 2.0(僅 Small) |

**偵測器**:Ultralytics 官方 `yolov8m-seg.pt`(sha256 51fa7e5e…)，放在 `checkpoints/`。

`data/`、`depth_models/`、`checkpoints/` 不在版本控制內。

## 資料

資料集不附在本 repo，請從原始來源取得並遵守各自的授權:
- **comma2k19**(MIT):https://huggingface.co/datasets/commaai/comma2k19 ，放在 `data/input/comma2k19/Chunk_*.zip`。
- **Argoverse 2 Sensor**(CC BY-NC-SA 4.0,只能用於研究):https://www.argoverse.org/ 。
- 海盛提供的台灣案例影片不公開。

## 主要程式

| 檔案 | 用途 |
|---|---|
| `tools/depth_dash_scale.py` | 方法一:車道線追蹤、在深度剖面上量虛線週期得到 k、自車速 |
| `tools/marking_geometry.py` | 方法 C:地平線、A、自車速 |
| `tools/depth_dash_multicar.py` | 偵測與追蹤、每台車的距離與速度(`measure` 方法一、`measure-c` 方法 C)、評分 |
| `tools/depth_backends.py` | 五個深度模型的統一介面 |
| `tools/c2k19_extract.py` | comma2k19 影格(ffmpeg)與真值(CAN、定位、雷達)，兩者分開放 |
| `tools/seal_predictions.py` | 封存預測檔 |
| `tools/run_v2.sh`、`tools/run_v2_av2.sh`、`tools/run_v3.sh`、`tools/run_v4.sh` | 盲測的預測步驟(後一版同時輸出前面各版) |
| `tools/v5_combine.py` | 第五版:尺度檢查與兩個方法的自車速平均 |
| `tools/v6_far_ruler.py`、`tools/v6_score.py` | 第六版:30 m 以外每台車自己的尺，及其評分 |
| `tools/band_bottom.py` | 只用畫面決定分析範圍下緣(引擎蓋、疊字) |
| `tools/c2k19_radar_check.py` | 評分前檢查雷達距離與原始 CAN 的差是否全為 2.70 m |
| `tools/c2k19_pool.py` | 只讀 zip 目錄列出盲測候選段，並查名稱是否出現過 |
| `tools/score_extra_v3.py`、`tools/score_extra_v4.py`、`tools/score_extra_v5.py` | 各版結果文件中的額外評分(關閘、真值 < 30 m、配對比較與符號檢定),封存後才執行 |
| `tools/hood_check_v5.py` | 第五版結果的事後檢查:關閘時引擎蓋被偵測成車的影響 |
| `tools/report/demo_*.py` | 示範影片 |

`tools/av1_*.py`、`tools/av2_*.py` 是 Argoverse 1/2 的下載、準備與評分;`tools/haisheng_manual_truth.py` 讀海盛的人工畫格法真值。

## 參考文獻

1. G. P. Stein, O. Mano, A. Shashua, "Vision-based ACC with a single camera: bounds on range and range rate accuracy,"
   IEEE Intelligent Vehicles Symposium, 2003, pp. 120–125. https://doi.org/10.1109/IVS.2003.1212895
2. T. N. Schoepflin, D. J. Dailey, "Dynamic camera calibration of roadside traffic management cameras for vehicle speed
   estimation," IEEE Transactions on Intelligent Transportation Systems, vol. 4, no. 2, pp. 90–98, 2003.
   https://doi.org/10.1109/TITS.2003.821213
3. Lin et al., "Depth Anything 3: Recovering the Visual Space from Any Views," arXiv:2511.10647, 2025.
   https://arxiv.org/abs/2511.10647 ;程式與模型 https://github.com/ByteDance-Seed/Depth-Anything-3
4. 道路交通標誌標線號誌設置規則第 182 條(車道線:線段長 4 公尺、間距 6 公尺)。
   https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=K0040014&flno=182
