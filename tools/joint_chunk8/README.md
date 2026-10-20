# 第 8 包共同盲測:真值前要 commit 的程式(原樣複製)

依共同登錄(`docs/JOINT_CHUNK8_PREREG.md`)追記 1 第 12、14、20 點與第 C 節、追記 2:以下程式在第 8 包任何真值產生之前 commit 進本 repo。
檔案逐位元組照抄本機原檔(程式裡的絕對路徑照舊),不改任何內容。

| 檔案 | 用途 | 本機原檔 | sha256(前 16 碼) |
|---|---|---|---|
| `rescore_settled.py` | 第 3.1 節依據:照定案規則重算第 7 包(中心欄 615 / 582) | `/mnt/d/depth_work/joint/chunk8/prereg_dev/rescore_settled.py` | `757395e7d92aef79` |
| `rescore_settled_7a_fix.py` | 追記 1 第 12 點:第 7 包前半改用正確的試跑資料夾 | `/mnt/d/depth_work/joint/chunk8/prereg_dev/rescore_settled_7a_fix.py` | `5f434646eae43814` |
| `rescore_lib.py` | 上兩支用的重算函式(讀本 repo 真值檔) | `/mnt/d/depth_work/decisions_20261018/g1/rescore_lib.py` | `36b643bebe701e74` |
| `rescore_own.py` | 不引用 rescore_lib 的獨立重算(真值照共用取法在記憶體內計算) | `/mnt/d/depth_work/audit_chunk8_20261019/dryrun/scripts/rescore_own.py` | `f892ff4b3c5b1999` |
| `rescore_own_7b.py` | 同上,第 7 包後半 | `/mnt/d/depth_work/audit_chunk8_20261019/dryrun/scripts/rescore_own_7b.py` | `03877c837ccf00b3` |
| `vp_per_drive_rule.py` | 追記 1 第 20 點:每趟消失點 | `/mnt/d/depth_work/joint/chunk8/addendum1/vp_per_drive_rule.py` | `2619a67e97222cbd` |
| `move_stale_5c.py` | 追記 2 第 5 點:中斷後移開沒有快取時寫出的衍生檔 | `/mnt/d/depth_work/joint/chunk8/recover_20261019/move_stale_5c.py` | `049386279b252408` |
| `check_5c_and_compare.py` | 追記 2 第 8 點:封存前核對(MoGe-2 量測新舊、兩版不用 k 的檔) | `/mnt/d/depth_work/joint/chunk8/preseal/check_5c_and_compare.py` | `48b4f68937c776a7` |
| `compare_ignore_paths.py` | 追記 2 第 8 點:路徑換成相同後再比一次 | `/mnt/d/depth_work/joint/chunk8/preseal/compare_ignore_paths.py` | `92acca3ec3782a73` |
| `own_seals.sh` | 追記 1 第 6 點、追記 3 第 3 點:本 repo 格式的 v9d、v9 封存 | `/mnt/d/depth_work/joint/chunk8/preseal/own_seals.sh` | `402c98ccdea3fb16` |
