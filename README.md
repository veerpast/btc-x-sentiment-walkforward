# X sentiment and BTC price forecasting

An end-to-end, daily Bitcoin experiment for the CrowdWisdomTrading assessment. The model uses **only historical BTC closes and X post sentiment**. Tavily news is collected for a separate current-outlook summary, never as a model feature. No API keys or private post exports belong in this repository.

## Run

Python 3.10+; install `pip install -r requirements.txt`. Copy `.env.example` to `.env` and set `APIFY_TOKEN` and `TAVILY_API_KEY` locally. The default actor is `apidojo/twitter-scraper-lite`; check its current input schema in Apify before invoking a paid run. To avoid actor schema drift or unexpected charges, the normal path is to export the actor's dataset as JSON and run:

```bash
python -m cwt.cli --posts data/apify_posts.json --asset BTC-USD --out results
```

The CLI fetches matching daily historical closes from Yahoo's chart API, stores normalized posts and bars in SQLite, writes forecasts, walk-forward predictions, metrics, a sentiment/price figure, and an outlook. It fails clearly when no posts or too few aligned observations are available. For reproducibility, optionally provide a price CSV with `date,close` using `--prices`; no network is then required. Historic tweet CSVs can be used with `--posts` when they contain at least date/timestamp and text columns. The 180-day recent scrape and historic evaluation are separate cohorts: use `--start YYYY-MM-DD --end YYYY-MM-DD` for a historical corpus with matching prices. Never claim an older dataset is a current scrape.

The repository also includes an **actual archival evaluation** from the public CrypTop12 tweet/price corpus. Run `python -m cwt.historical` to regenerate `results/historical/evaluation.md`, metrics, forecasts and chart from `data/historical_daily.csv`. The CSV is a derived daily aggregate, not original posts; its source and timing assumptions are documented in the evaluation. It covers 2017–2018 and must not be presented as a current 180-day result.

Run `python scripts/build_historical_daily.py` to reconstruct the aggregate from pinned public source files. It downloads the archival daily posts and price series, converts IST timestamps to UTC, deduplicates post IDs, and applies the documented sentiment and engagement rules. It requires network access and does not fetch current posts.

The committed [evaluation](reports/evaluation.md) reports that this simple model **underperformed the no-change baseline at all five horizons**. Its idealized one-day signal diagnostic also lost money in this archival window. This is useful negative evidence, not a validated trading signal.

## Data contract

Posts are filtered for Bitcoin cashtag/ticker/keyword matches and deduplicated by ID or stable content hash. UTC timestamps are assigned to UTC calendar days for a 24/7 asset; the ingestion code also derives America/New_York timestamps for inspection. A daily close is available only **after** its UTC bar ends. For day `t`, features use posts dated no later than `t` and closes through `t`. A forecast issued after that close has `n+0` equal to the observed close (a reconstruction, not a prediction); horizons `n+1` through `n+5` are forward closes.

Sentiment uses a small transparent finance lexicon, engagement weight capped to avoid one viral post dominating, daily post count, weighted mean, and lagged/rolling sentiment and price returns. Missing-post days get zero volume and neutral sentiment; missing prices are never forward filled. There are no news, volume, on-chain, or future columns in model inputs.

Walk-forward splits refit on past observations only. For horizon `h`, training ends at least `h` bars before each test prediction so all training labels would be known at issue time. An expanding initial window of 60 observations is required. Models are ridge regressions on returns with a naive zero-return baseline and a price-only comparison. A one-day long/cash diagnostic follows the sign of the out-of-sample `n+1` forecast with 10 bps per position change. Its same-close fill is generally unavailable after the signal is known. Sharpe, Sortino, and maximum drawdown are idealized signal metrics, not executable trading results.

## Apify and Tavily

`python -m cwt.cli --collect-apify ...` uses the configured actor via the Apify API and writes the resulting dataset locally. Actor input schemas vary; the CLI accepts a JSON actor input file with `--actor-input` so the exact validated configuration is reproducible. The default input is intentionally conservative (`searchTerms`, `maxItems`) and should be adjusted to the selected actor. A failed or empty actor run stops the pipeline. `--news` fetches recent Tavily results for a qualitative outlook only. Keys are read from environment variables, never printed or committed. Actor ID and input JSON can be reported without disclosing credentials; reviewers can use their own API keys.

## Limits

The browser session's first Apify run on 5 November 2026 returned **zero items**, so it supplies no recent training data. A recent 180-day evaluation can be produced only after a real, aligned scrape is supplied. The committed archival evaluation is explicitly dated. Historical tweet datasets cited in the brief mostly predate the requested 180 days. Sparse posts, evolving platform access, lexicon error, multiple horizon selection, transaction costs, and daily bar timing can materially change results.

## Outputs

`results/metrics.json`, `results/evaluation.md`, `results/forecasts.csv`, `results/walk_forward.csv`, `results/sentiment_price.svg`, `results/outlook.md`, and `results/data.sqlite`. CSVs and database are local run artifacts. The report documents provenance, exact date span, sample counts, model/baseline error, and strategy metrics. Run `python -m unittest discover -s tests` for deterministic timing and metric checks.
