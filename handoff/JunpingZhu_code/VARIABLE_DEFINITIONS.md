# Variable Definitions and Trade Alignment

## Current behavior
- If `--raw_trade_path` is provided, the pipeline now uses a GitHub-style trade-anchor path.
- Each trade is aligned to the most recent quote for the same `option_ticker` using backward as-of matching.
- If no trade file is provided, the old quote-anchor fallback path is still used.

## Trade alignment
- Anchor event: one option trade
- Alignment rule: exact `option_ticker` match, then nearest quote with `quote_ts <= trade_ts`
- Current-state variables are taken from that aligned quote.

## Label definitions
- `mid_avg` / `trade_avg` in trade-anchor mode both use:
  - `future_avg_trade_price / current_mid - 1`
- `future_avg_trade_price` is the equal-weight mean of future trade prices in `(t, t+h]`
- `n_future_quotes` is kept for output compatibility and stores future trade count in trade-anchor mode

## Current state variables
- `current_mid`: aligned quote mid `(bid + ask) / 2`
- `current_relative_spread`: aligned quote relative spread `(ask - bid) / mid`
- `current_bid_size`: aligned quote best bid size
- `current_ask_size`: aligned quote best ask size
- `current_quote_size_sum`: `bid_size + ask_size`
- `current_lob_imbalance`: `(ask_size - bid_size) / (ask_size + bid_size)`
- `strike`: option strike
- `DTE`: days to expiration
- `option_type_flag`: call=1, put=0

## Window definitions
- `window_0_5s_*`: statistics over past trades in `(t-5s, t]`
- `window_5_10s_*`: statistics over past trades in `(t-10s, t-5s]`
- All window features are computed by exact `option_ticker` match on the trade timeline.

## Trade-anchor window variables
- `breadth`: number of trades in the lookback window
- `immediacy`: window length in seconds divided by trade count
- `volume_all`: sum of trade size
- `volume_avg`: average trade size
- `volume_max`: maximum trade size
- `mid_lambda`: `(mid_last - mid_first) / volume_all`
- `lob_imbalance`: window mean of aligned quote LOB imbalance
- `txn_imbalance`: `sum(trade_size * trade_direction) / volume_all`
- `past_return`: `1 - avg_trade_price / mid_last`
- `turnover`: `volume_all / current_quote_size_sum`
- `mid_autocov`: mean of adjacent log trade-return products
- `quoted_spread_mean`: window mean of aligned quote relative spread
- `effective_spread`: `sum(log(trade_price / current_mid) * trade_direction * trade_size * trade_price) / sum(trade_size * trade_price)`
- `no_quote_flag`: 1 if the trade window is empty
- `no_trade_flag`: 1 if the trade window is empty

## Trade direction proxy
- Primary rule: `trade_price > current_mid` gives `+1`, `trade_price < current_mid` gives `-1`
- Fallback rule when `trade_price == current_mid`: use tick direction from the last non-zero trade-price change

## Notes
- This is intentionally closer to the GitHub implementation than the earlier quote-anchor handoff version.
- The turnover denominator is approximated with aligned displayed quote depth rather than an external base.
- The quote-anchor path is still available as a fallback for quote-only runs.
