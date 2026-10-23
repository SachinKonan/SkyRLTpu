# Tier-2 floorplan check (2026-10-23)

Each method's repeat-1 NG45 placement, run through the challenge's `evaluate_with_orfs.py` (996193c,
plus `name_lookup.patch`) with the ORFS image openroad/orfs:26Q2-385-gbd2a6b695 up to floorplan,
default 12 um spacing push (ORFS_MIN_GAP_UM unset). Logs: placement-fair-eval/orfs/fptest/.

`name_lookup.patch`: without it no macro is found in ODB (OpenROAD names these Genus instances
`sram_block\[0\]...`; the script's candidates were double-quoted, so Tcl dropped the backslashes),
every macro stays unplaced and PDN fails for every method. Same failure in the 26Q2-169 (Apr 23) image.

| Method   | ariane133 | ariane136 | mempool_tile | nvdla | PDN passes |
|----------|-----------|-----------|--------------|-------|------------|
| AbuPlace | OK        | OK        | OK           | OK    | 4/4        |
| ArchGen  | PDN-0179  | PDN-0179  | OK           | OK    | 2/4        |
| MOMO fill| PDN-0179  | PDN-0179  | OK           | PDN-0179 | 1/4     |
| MOMO 300 | PDN-0179  | PDN-0179  | OK           | PDN-0179 | 1/4     |

PDN-0179: "Unable to repair all channels" (power straps cannot pass the gaps between macros).
Mean spacing-push displacement: AbuPlace 0.10-0.16 um; ArchGen 3.4-18.5 um; MOMO 9.5-35.6 um.
Only the passing placements (plus ORFS's rtl_macro_placer reference) go on to the full flow.
