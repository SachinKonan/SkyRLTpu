# Portfolio v2 public-data source audit

Checked 2026-09-13. This started as a source shortlist; the actual v2 build uses
the defeatbeta snapshot, FNSPID headlines and the two Kaggle archives below.
Existing v1 model comparisons retain their original data and prompts.

## Agreed periods

- Inner train: 2020-2023.
- Inner validation: 2024.
- Discovery validation: 2025; this score may feed discovery.
- Final evaluation: available completed 2026 trading days; no search feedback.
- Late-2019 observations provide causal feature/context warm-up.

## Sources inspected

| Source | Pinned revision or observed metadata | Potential role |
|---|---|---|
| [defeatbeta/yahoo-finance-data](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data) | `9ab94c890bf72f574bccd10fe89ee10a4227d26d`; ungated; spec records prices updated 2026-09-13 05:21:53 UTC | Bulk daily prices, separate dividends/splits, Treasury yield observations |
| [paperswithbacktest/Stocks-Daily-Price](https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price) | `9d5a40219514227e5a754deef3b5a6e41571d22b`; ungated; card advertises 7,764 symbols through 2026-08-05 | Alternative history with adjusted closes; audit coverage and adjustment conventions |
| [Arandkei historical delisted archive](https://www.kaggle.com/datasets/rodas86/arandkei-historical-delisted-assets-archive) | Search-visible card advertises a growing 22 MB archive; files not downloaded or validated | Recover selected missing delisted histories, subject to provenance and terminal-event checks |
| [yfinance](https://github.com/ranaroussi/yfinance) | Open-source downloader; direct source queries still needed | Retrieve selected histories/actions and refresh dates absent from a frozen mirror |

Repository update times do not establish the last trading date for every asset.
Yahoo-derived mirrors are not independent market-data verification. Public
availability and a mirror's license tag do not establish upstream redistribution
rights. Dataset contents and licensing still require source-specific review.

## Findings from downloaded files

- The pinned defeatbeta prices reach 2026-09-11 for 29 of the selected historical
  Dow identities. WBA ends with a zero-volume synthetic quote on 2025-08-28;
  that quote is not treated as a traded price. WBA is retained with missing-price
  rows afterward. UTX's vendor history is stored under its successor ticker RTX.
- The actual defeatbeta news file has 1,325,033 stock-linked rows, all in 2025 or
  2026, and every report_date has day precision only. A repository update is not
  a publication timestamp. All selected headlines use reported date + 2 days
  at 00:00 UTC as a conservative availability convention.
- The Papers With Backtest card states that downloads require a subscription,
  despite the API metadata returning gated=false. The card also acknowledges
  substantial missing delisted coverage. This source was not used in the build.
- Historical stock-linked headlines were projected from
  [beachside1234/FNSPID](https://huggingface.co/datasets/beachside1234/FNSPID),
  revision `749c61187fb353c62c91a36f775a909de8c1ecb2`. The selected 2019-2023
  extracts contain 109,485 Nasdaq rows and 17,179 external-source rows before
  cross-source deduplication.
- [Kaggle Polygon sample](https://www.kaggle.com/datasets/rdolphin/financial-news-with-ticker-level-sentiment),
  version 1, and [Kaggle Apple archive](https://www.kaggle.com/datasets/frankossai/apple-stock-aapl-historical-financial-news-data),
  version 1, supplement coverage. Only original titles, dates, URLs and stock
  associations are used; supplied model-generated sentiment and summaries are
  excluded. These archives do not establish complete 2024 coverage for all stocks.
- None of these downloaded sources proves the original as-published version
  of subsequently edited text. The model card distinguishes this source-vintage
  limitation from the tested temporal guarantees of our own joins.

## Assembly rules

Select a modest universe using only information available before 2020. Do not
drop a selected security because it later delisted or has inconvenient missing
data. Record unavailable securities and unresolved terminal payouts explicitly.
Maintain stable security identities through ticker changes and ticker reuse.

Use a primary history per security and reconcile overlapping dates before any
source substitution. Preserve raw inputs, source revisions, download dates and
checksums. Reconstruct corporate actions consistently; do not directly splice
adjusted prices from snapshots taken at different dates. Reject duplicate dates,
invalid prices and unexplained jumps. Missing prices, exchange holidays, halts
and delisting are distinct states, not generic forward-fill opportunities.

Inspect coverage and input integrity in 2026 without evaluating candidate
performance to choose the universe, preprocessing or algorithms. Keep its
return labels outside discovery workers. Freeze candidate selection before
the final evaluation.

Only deploy the compact selected panel, not the multi-million-row source lake.
Keep four CPU cores and 8 GiB as the initial per-candidate budget and remeasure
fit/replay time. The v1 complete-panel builder needs changes for tradability
masks, corporate actions and delisting handling before claiming this v2 design
is implemented.
