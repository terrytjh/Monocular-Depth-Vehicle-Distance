# Monocular-Depth-Vehicle-Distance

**Monocular Vehicle Distance Measurement Based on AI Depth Estimation Models**

**以 AI 深度估算模型為基礎的單鏡頭影像行車距離量測方法**

從一台行車紀錄器的影片,量出**前車與左右車道車輛的距離**、**這些車的相對速度與絕對速度**,以及**自車速**。
每一個速度都附 95% 範圍。用到的只有影片本身、公開的深度模型與偵測器,以及道路上的法定標線。

## 兩個方法

**方法一:深度模型 × 法定虛線尺**(主方法)
單目度量深度模型(Depth Anything 3 metric)給每個像素一個距離,但它的公尺尺度會隨相機與場景偏掉。
車道虛線的週期是法規定的(台灣第 182 條 10 m;加州高速公路 Caltrans A20A 14.63 m;美國聯邦 MUTCD 12.19 m),
所以在**模型自己的深度裡**量出虛線週期 P,就得到尺:`k = 法定週期 ÷ P`,距離 = k × 模型讀值。
自車速是同一條深度剖面上,虛線邊緣在兩張影格之間的平移。

**方法二:方法 C,只用標線幾何**(對照方法,不用深度模型)
平路、針孔相機:畫面第 y 列的路面距離 `d = A / (y − y_h)`。地平線 y_h 取兩條車道線的交點;
A 由虛線在 `u = 1/(y − y_h)` 上的週期換算(`A = 法定週期 ÷ 週期`)。

兩個方法用同一組車輛框與追蹤編號(YOLOv8m-seg + BoT-SORT),速度、95% 範圍與評分也用同一套程式,
差別只在距離與自車速怎麼來。

## 驗證方式:先登錄、再預測、封存、最後才看答案

1. 方法與參數只在「答案已經開過」的開發片上調整,每次看答案都記在 `docs/TERRY_DEV_LOG.md`。
2. 方法定版後寫成事前登錄(`docs/DEPTH_DASH_V2_PREREG.md`)並 commit;commit 的時間就是登錄時間。
3. 在答案從沒被讀過的盲測片上預測(`tools/run_v2.sh`、`tools/run_v2_av2.sh`),程式只讀影片。
4. `tools/seal_predictions.py` 把全部預測檔的 sha256 封存,並確認真值檔還不存在。
5. 封存之後才產生真值、評分;評分程式先核對封存雜湊。一批盲測片只用一次。

真值:comma2k19 的原廠雷達、CAN 與定位(`tools/c2k19_extract.py`);Argoverse 2 的光達 3D 框與定位。

## 結果

(盲測完成後填入。)

## 安裝

```bash
python3 -m venv ~/venvs/depthbench && source ~/venvs/depthbench/bin/activate
pip install torch ultralytics opencv-python numpy pandas pyarrow scipy lap openpyxl pillow matplotlib moviepy
```
測試環境:Python 3.13、torch 2.14、ultralytics 8.4.153、opencv-python 4.13。

**Depth models**:官方 repo 放在 `depth_models/`,權重照各 repo 的說明下載:

| 模型 | repo | 測試用的 commit | 授權 |
|---|---|---|---|
| Depth Anything 3(`da3_metric`,主方法) | https://github.com/ByteDance-Seed/Depth-Anything-3 | 3d835ec | Apache 2.0 |
| Metric3D v2 | https://github.com/YvanYin/Metric3D | eb5b6fa | BSD-2(權重授權未載明) |
| UniDepth v2 | https://github.com/lpiccinelli-eth/UniDepth | 8d8cfe4 | CC BY-NC 4.0 |
| Depth Pro | https://github.com/apple/ml-depth-pro | 9e65e4d | Apple AMLR |
| Depth Anything V2(Small) | https://github.com/DepthAnything/Depth-Anything-V2 | a561b84 | Apache 2.0(僅 Small) |

**偵測器**:Ultralytics 官方 `yolov8m-seg.pt`(sha256 51fa7e5e…),放在 `checkpoints/`。

`data/`、`depth_models/`、`checkpoints/` 不在版本控制內。

## 資料

資料集不附在本 repo,請從原始來源取得並遵守各自的授權:
- **comma2k19**(MIT):https://huggingface.co/datasets/commaai/comma2k19 ,放在 `data/input/comma2k19/Chunk_*.zip`。
- **Argoverse 2 Sensor**(CC BY-NC-SA 4.0,只能用於研究):https://www.argoverse.org/ 。
- 海盛提供的台灣案例影片不公開。

## 主要程式

| 檔案 | 用途 |
|---|---|
| `tools/depth_dash_scale.py` | 方法一:車道線追蹤、在深度剖面上量虛線週期得到 k、自車速 |
| `tools/marking_geometry.py` | 方法 C:地平線、A、自車速 |
| `tools/depth_dash_multicar.py` | 偵測與追蹤、每台車的距離與速度(`measure` 方法一、`measure-c` 方法 C)、評分 |
| `tools/depth_backends.py` | 五個深度模型的統一介面 |
| `tools/c2k19_extract.py` | comma2k19 影格(ffmpeg)與真值(CAN、定位、雷達),兩者分開放 |
| `tools/seal_predictions.py` | 封存預測檔 |
| `tools/run_v2.sh`、`tools/run_v2_av2.sh` | 盲測的預測步驟 |
| `tools/report/demo_*.py` | 示範影片 |

`tools/av1_*.py`、`tools/av2_*.py` 是 Argoverse 1/2 的下載、準備與評分;`tools/haisheng_manual_truth.py` 讀海盛的人工畫格法真值。
