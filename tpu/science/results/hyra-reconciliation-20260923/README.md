# Hyra qubit-routing reconciliation — September 23, 2026

The published counts reconcile exactly with the cached 72-case benchmark when
its SABRE costs are retained for the 11 cases without exported Hyra circuits.
This is an accounting reconstruction, not a claim that Hyra documented this
fallback or supplied 72 independently verified circuits.

| Topology | Exported cases | Omitted cases | Verified exported SWAPs | Omitted SABRE SWAPs | Accounted full-suite SWAPs | Published added CNOTs |
|---|---:|---:|---:|---:|---:|---:|
| Q20 | 16 | 8 | 12,622 | 1 | 12,623 | 37,869 |
| Heron | 22 | 2 | 41,534 | 0 | 41,534 | 124,602 |
| Willow | 23 | 1 | 31,966 | 0 | 31,966 | 95,898 |
| Total | 61 | 11 | 86,122 | 1 | 86,123 | 258,369 |

## What was verified

- Pinned Tencent's source files to commit `bcda2e29200fbeaeb1620acca0670cc445e8688f`.
- All 61 IDs are unique and belong to the identical cached benchmark suite
  (`84bc454b7505875a01a72e7f9ffe033185d273a2e7dddf3544dc89d7bc27f7cf`).
- Independently replayed all 61 circuits with the cached SimpleTES verifier and
  Qiskit 1.2.0: 61/61 passed initial-mapping, topology-adjacency, and circuit
  equivalence checks. Parsed actual SWAP operations rather than trusting a
  reported count. Slurm CPU job14322527; no TPU workloads were modified.
- The 11 omitted IDs have ten zero-SWAP SABRE baselines and one one-SWAP baseline,
  `alu-v0_27_q20`. Adding their cached costs exactly reproduces each published
  topology total. The README does not explicitly specify its omission policy.

## Interpretation

The 61-versus-72 mismatch does not by itself show an inflated improvement:
its entire numerical difference is one SWAP. For a full-suite table, label Hyra
as "61 exported circuits plus SABRE costs for 11 omitted cases". Do not call
this a fresh 72/72 Hyra replay. The directory supplies routed circuit outputs,
not a routing-policy implementation that this audit could rerun on all inputs.
Search budget, model budget, and generic-policy versus circuit-output discovery
are not established as equivalent by this audit.

The reconstructed 72-case weighted reward under our reward formula is
0.5558786131831626. Its 86,123 SWAP total is 4.0% below cached SimpleTES-120B
(89,679), and 1.4% below cached SimpleTES-Gemini (87,347). These cached references
were not fetched or independently rerun during this reconciliation.

## Evidence

- `verification.json`: every independently checked case, verifier/source hashes,
  omission accounting, and reconstructed totals.
- `per-case.csv`: 72-case counts, with explicit exported/fallback provenance.
- `omitted-cases.csv`: the 11 omitted IDs and cached SABRE/LightSABRE costs.
- `provenance.json`: immutable source URLs, hashes, and verification environment.
- `audit.py`: repeatable independent validation; takes explicit input paths.
- Downloaded input: `.science/hyra-routing-audit-20260923/solution.json`.

Source: https://github.com/Tencent-Hunyuan/Hyra-results/tree/bcda2e29200fbeaeb1620acca0670cc445e8688f/AI4Science/qubit_routing
