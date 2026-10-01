# Monocular-Depth-Vehicle-Distance

**Monocular Vehicle Distance Measurement Based on AI Depth Estimation Models**

**以 AI 深度估算模型為基礎的單鏡頭影像行車距離量測方法**

從一台行車紀錄器的影片，量出**前車與左右車道車輛的距離**、**這些車的相對速度與絕對速度**，以及**自車速**。
每一個速度都附 95% 範圍。用到的只有影片本身、公開的深度模型與偵測器，以及道路上的法定標線。

## 兩個方法

**方法一:深度模型 × 法定虛線尺**(主方法)
單目度量深度模型(Depth Anything 3 metric)給每個像素一個距離，但它的公尺尺度會隨相機與場景偏掉。
車道虛線的週期是法規定的(台灣第 182 條 10 m;加州高速公路 Caltrans A20A 14.63 m;美國聯邦 MUTCD 12.19 m)，
所以在**模型自己的深度裡**量出虛線週期 P，就得到尺:`k = 法定週期 ÷ P`，距離 = k × 模型讀值。
自車速是同一條深度剖面上，虛線邊緣在兩張影格之間的平移。

**方法二:標線幾何法，不用深度模型**(對照方法;程式與早期文件中稱「方法 C」)
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
| 第三版 | 2026-10-01 15:03(`v3-prereg`) | **只改一項**:深度法自車速改用前後 ±2.5 秒的局部尺(含保護)，其餘同第二版 | comma2k19 Chunk_2 194 段(另一台車，答案從未讀過) | 預測中;第二、三版在同一批片上同時輸出，兩版都會報告 |

文件:`docs/DEPTH_DASH_V2_PREREG.md`、`docs/DEPTH_DASH_V2_RESULTS.md`、`docs/DEPTH_DASH_V3_PREREG.md`;
開發過程(每次看答案的時間與內容)`docs/TERRY_DEV_LOG.md`;封存雜湊 `docs/seals/`。

## 結果(第二版盲測，`docs/DEPTH_DASH_V2_RESULTS.md`)

comma2k19 白天高速公路 95 段，真值為原廠雷達與定位;事前寫下的 6 條預期全部達成。

| | 深度 × 虛線尺 | 標線幾何法 |
|---|---|---|
| 有輸出的段(其餘拒發) | 73 / 95 | 80 / 95 |
| 本車道 / 左右一道距離，中位誤差(偏差) | **5.7%(+0.3%)/ 6.3%(−0.6%)** | 16.2%(−16.1%)/ 17.5%(−17.4%) |
| 他車相對速度 MAE(全部猜 0) | 2.04 km/h(6.86) | 1.92 km/h(6.55) |
| 他車絕對速度 MAE | 5.31 km/h | 3.88 km/h |
| 自車速每秒 MAE | 5.81 km/h | 4.82 km/h |
| 95% 範圍實際覆蓋率(相對 / 絕對 / 自車) | 94 / 95 / 95% | 94 / 93 / 92% |

- **尺的作用**:同一個深度模型不乘尺時，在焦距與名目值相差大的相機上(Argoverse 2 一支開發片，13 筆配對，封存後的描述性對照)距離誤差 32%，乘上虛線尺後 5%。
- **主要誤差來源**:深度法的自車速(模型的公尺尺度在數秒內漂移)，它直接影響他車絕對速度 → 第三版只改這一項。
- 白天一般道路與夜間另外報告，拒發多;詳見結果文件。

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
| `tools/run_v2.sh`、`tools/run_v2_av2.sh`、`tools/run_v3.sh` | 盲測的預測步驟(第三版同時輸出第二版) |
| `tools/report/demo_*.py` | 示範影片 |

`tools/av1_*.py`、`tools/av2_*.py` 是 Argoverse 1/2 的下載、準備與評分;`tools/haisheng_manual_truth.py` 讀海盛的人工畫格法真值。
