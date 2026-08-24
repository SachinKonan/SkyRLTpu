# Known issues

## audit_programs.classify() is Python/erdos-specific  (found 2026-08-05, mid-pilot)
`uses_base = "initial_h_values" in text` never matches C++ sources, so for fc46/fc302 the
SUSPECT branch is unreachable and any weak signal escalates to HARDCODED -> the program is
DISCARDED from the store. Observed 2/100 false positives in fc46 round 4 (both below the
running best, so no effect on the headline).

NOT patched during the fc46 pilot on purpose: the audit runs identically in all three arms, so
leaving it keeps the arm comparison internally valid. Patching mid-run would treat arm 1
differently from arms 2-3 and confound the result.

FIX BEFORE THE ERDOS RUN (where the check genuinely matters -- erdos is memorisable and Python):
  - make classify() language-aware: pass `lang`
  - for cpp, the threat model is different (the instance arrives on stdin at grading time, so a
    fixed answer cannot be embedded). Treat "reads stdin" (cin/scanf/getline) as the analog of
    `uses_base`, and only flag cpp on explicit embedded-data decoding (base64/zlib/frombuffer).
