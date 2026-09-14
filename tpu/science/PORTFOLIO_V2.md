# Portfolio v2: joined daily prices and headlines

The v2 data build uses a fixed historical equity universe, recent public price
data, historical headlines and an explicit temporal join. The original v1
Ray model pilots keep their original data. This document describes the new
data layer; its dataframe interface is not interchangeable with the v1 grader's
NumPy-only contract.

The verified release is available at `.science/data/portfolio-v2`. Its
[model card](data_cards/portfolio-v2/DATA_CARD_MODEL.md),
[coverage table](data_cards/portfolio-v2/coverage.csv), and
[verification evidence](data_cards/portfolio-v2/README.md) are retained here
alongside the code. The dataframe has 58,020 rows and 204,982 stock-linked
headline records; the latter count is not a count of globally unique articles.

The dataset directory contains:

- `joined.parquet`: one row per exchange session and historical equity identity,
  including both backward-looking inputs and explicitly named forward labels.
  This is a trusted research artifact, never an entire candidate input.
- `news.parquet`: deduplicated titles, source dates, conservative availability
  timestamps, stock associations, URLs and source identifiers.
- `coverage.csv`: per-equity/per-year price, feature, label and news counts.
- `candidate/train.parquet` and `candidate/valid.parquet`: fitting inputs and
  return labels that mature within their respective inner periods.
- `trusted/discovery_observations.parquet` and `trusted/final_observations.parquet`:
  sequential evaluator inputs. These full future tables must remain outside
  the candidate process; send only the current causal observation.
- `trusted/*_labels.parquet`: independent evaluator labels and validity flags.
- `DATA_CARD_MODEL.md`: prompt-ready schema, equity names, pre-final coverage,
  timestamp conventions, allowed libraries and resource limits.
- `DATA_CARD_AUDIT.md`: researcher card including 2026 coverage.
- `manifest.json` and `verification.json`: provenance and executed checks.

## Read the joined dataframe

Run this in the trusted preparation/research environment:

```python
from pathlib import Path
import pandas as pd
from tpu.science.portfolio_v2 import fit_frame, observation

root = Path('.science/data/portfolio-v2')
joined = pd.read_parquet(root / 'joined.parquet')
train = fit_frame(joined, 'train')
valid = fit_frame(joined, 'valid')
obs = observation(joined, '2025-01-02')
```

`observation()` uses a fixed allowlist; it never returns execution timestamps,
forward returns, future label-validity masks or raw retrospectively adjusted
price levels. `fit_frame()` rejects discovery/final and masks labels crossing
the inner-period boundary. NaN is an unavailable value, not zero return.

For a dataframe-based `fit(train, valid, rng, *, time_budget_s)` implementation,
each row is a stock/session example. Fit text vocabularies and normalization
using the permitted fitting inputs. A subsequent evaluator can supply `obs`
plus drifted holdings to `act`; all temporal slicing belongs in the trusted
evaluator. Do not run that evaluator by handing a policy the full joined file.

## Rebuild

Install the separate trusted preparation environment. Data preparation packages
such as DuckDB, PyArrow and exchange-calendars do not expand the candidate's
library allowlist.

```bash
uv venv .science/data-venv --python .science/venv/bin/python
uv --no-config pip sync --python .science/data-venv/bin/python tpu/science/requirements-data.lock
.science/data-venv/bin/python -m tpu.science.fetch_public_data \
  .science/data/portfolio-v2-raw --news-history
srun -p cpu -c 4 --mem=12G --time=00:15:00 \
  .science/data-venv/bin/python -m tpu.science.extract_news_history \
  .science/data/portfolio-v2-raw/fnspid \
  --local .science/data/portfolio-v2-raw/fnspid-downloads
srun -p cpu -c 4 --mem=12G --time=00:20:00 \
  .science/data-venv/bin/python -m tpu.science.build_portfolio_v2 \
  --raw .science/data/portfolio-v2-raw --output .science/data/portfolio-v2-new
srun -p cpu -c 4 --mem=12G --time=00:20:00 \
  .science/data-venv/bin/python -m tpu.science.verify_portfolio_v2 \
  --raw .science/data/portfolio-v2-raw --output .science/data/portfolio-v2-new
```

Downloads happen on a network-enabled host. CPU compute nodes in this workspace
could not resolve Hugging Face, so the extraction phase uses local pinned files.
The builder refuses to overwrite an output directory. Download revisions and
Kaggle versions are pinned; changing them creates a new dataset version and
requires repeating the coverage and causality checks.

## Scientific limits

The builder prevents future rows from entering inputs and does not fill missing
prices. Retrospective source revisions, unavailable original headline versions,
and incomplete news collection are explicitly documented rather than claimed
away. Empty news lists are not a verified zero-event count. Price-return labels
use the vendor's historical adjustment convention and are not a complete
corporate-actions ledger. In particular, a delisting or unobserved execution
price yields an unavailable label, not an invented liquidation payout.

Before scoring portfolios with this new data, the evaluator must implement a
pinned treatment of corporate actions, unavailable execution prices and holdings
in delisted names. The v1 grader must not silently consume these new labels or
skip missing-return rows. Data quality audits do not substitute for that work.
