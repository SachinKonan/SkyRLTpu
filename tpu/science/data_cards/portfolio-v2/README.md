# Verified price/headline dataframe

This release contains 58,020 equity/session rows for a fixed historical
30-equity universe, from January 2, 2019 through September 11, 2026. It joins
204,982 deduplicated stock-linked headline records. An article associated with
multiple equities can appear more than once in that count.

The stable local data directory is `.science/data/portfolio-v2`, pointing to
the immutable `portfolio-v2-build003` directory. Read `joined.parquet` with
pandas in the trusted data environment; it is approximately 22 MB on disk and
85 MiB as a dataframe. It includes forward labels and is not a candidate input.
See the [usage and rebuild guide](../../PORTFOLIO_V2.md).

- [Model-facing card](DATA_CARD_MODEL.md): equity names, 2019–2025 coverage,
  input features, causal timing, allowed CPU libraries and provisional budgets.
- [Researcher card](DATA_CARD_AUDIT.md) and [coverage table](coverage.csv):
  all years, including 2026. Keep these out of candidate prompts.
- [Manifest](manifest.json): pinned sources, schema, splits and build hashes.
- [Temporal verification](verification.json): 986,582 news links checked,
  four prefix reconstructions, split-boundary purging and market-calendar checks.
- [Release checks](release_checks.json): all 15 source-file hashes, construction
  hashes, card/coverage consistency and discovery/final export consistency.
- [Label checks](label_checks.json): every forward label agrees with its price
  and dividend inputs, unavailable targets remain null, all numeric fields have
  no infinities, and every session has all 30 equity identities.
- [Test results](tests.xml): 23 tests passed, including eight temporal tests.
- [Release checksums](release.json): all data files, cards and audit records.

The split is 2020–2023 training, 2024 inner validation, 2025 discovery feedback,
and 2026 final evaluation, with 2019 warm-up. No final-test portfolio score was
used to build or select this release.

These checks establish pipeline causality. Retrospective public archives do
not certify original headline versions or unrevised historical prices. News
coverage is particularly sparse in 2024. Unknown execution prices and terminal
outcomes remain null; this is not a complete corporate-actions ledger.

The dataframe and cards are ready. Integration with the Ray portfolio scorer,
explicit corporate-action/delisting treatment, and CPU timing of candidate text
learners remain separate work. Existing model-pilot scores use portfolio data
v1, even when their prompt is called version 2.
