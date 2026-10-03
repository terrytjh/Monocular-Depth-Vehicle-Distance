# Seal manifests

Copies of the seal files written by `tools/seal_predictions.py` before any truth was produced or read. Each lists the
sha256 of every prediction file, the git commit at the time of sealing, and the truth files confirmed absent. The
originals stay next to the predictions (`data/output/dash_scale/<batch>/seal.json`, outside version control); these copies
are here so the hashes are on record. The scorers refuse any prediction whose hash differs from its seal.

| file | batch | sealed (UTC) | prediction files |
|---|---|---|---|
| `blind_v2_seal.json` | second version, comma2k19 149 blind segments + AV2 four logs (`docs/DEPTH_DASH_V2_PREREG.md`) | 2026-09-30 17:13:58 | 651 |
| `heldout_hs_003-005_seal.json` | Haisheng 20251230 003-005, held-out (addendum 1 of the second version) | 2026-09-30 18:36:16 | 6 |
| `heldout_hs1029_seal.json` | Haisheng 20251029 eight car cases, held-out (`docs/DEPTH_DASH_V3_PREREG.md` 3.2); case numbers replaced by codes A-H, hashes unchanged, sha256 of the original file recorded inside | 2026-10-01 07:09:35 | 17 |
| `blind_v3_seal.json` | third version, comma2k19 Chunk_2 194 blind segments (`docs/DEPTH_DASH_V3_PREREG.md`) | 2026-10-01 14:59:43 | 969 |
| `blind_v4_tw_seal.json` | fourth version, 18 Taiwanese web clips, ego speed only (`docs/DEPTH_DASH_V4_PREREG.md` 2.2) | 2026-10-02 02:06:14 | 86 |
| `blind_v4_seal.json` | fourth version, comma2k19 Chunk_4 205 segments (`docs/DEPTH_DASH_V4_PREREG.md` 2.1) | 2026-10-02 06:37:54 | 1155 |
| `blind_v5_seal.json` | fifth version, comma2k19 Chunk_5 211 segments (`docs/DEPTH_DASH_V5_PREREG.md`) | 2026-10-02 17:10:01 | 1381 |
| `blind_v6_seal.json` | sixth version, the same 211 segments, sealed after the fifth and before any truth (`docs/DEPTH_DASH_V6_PREREG.md`) | 2026-10-02 17:11:36 | 164 |
