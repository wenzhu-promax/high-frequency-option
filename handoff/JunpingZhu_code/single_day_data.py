"""Quote/trade loading and cleaning for the single-day pipeline."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EASTERN_TZ = "America/New_York"
TICKER_PATTERN = re.compile(
    r"^O:(?P<underlying>[A-Z.]+)(?P<expiration>\d{6})(?P<option_type>[CP])(?P<strike>\d{8})$"
)


def load_sp100_symbols(path):
    symbols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            symbol = line.strip()
            if symbol:
                symbols.append(symbol)
    print(f"Loaded {len(symbols)} SP100 symbols from {path}", flush=True)
    return set(symbols)


def detect_columns(columns):
    candidates = {
        "ticker": ["ticker", "symbol", "option_ticker"],
        "timestamp": ["sip_timestamp", "timestamp", "ts", "quote_timestamp"],
        "bid": ["bid_price", "bid", "best_bid"],
        "ask": ["ask_price", "ask", "best_ask"],
        "bid_size": ["bid_size", "bidsize", "bid_sz", "best_bid_size"],
        "ask_size": ["ask_size", "asksize", "ask_sz", "best_ask_size"],
    }
    found = {}
    lower_map = {str(col).lower(): col for col in columns}
    for key, names in candidates.items():
        for name in names:
            if name.lower() in lower_map:
                found[key] = lower_map[name.lower()]
                break
    missing = [key for key in candidates if key not in found]
    if missing:
        print("Could not auto-detect required columns.", flush=True)
        print(list(columns), flush=True)
        raise ValueError(f"Missing required logical columns: {missing}")
    print("Detected columns:", found, flush=True)
    return found


def detect_trade_columns(columns):
    candidates = {
        "ticker": ["ticker", "symbol", "option_ticker"],
        "timestamp": ["sip_timestamp", "timestamp", "ts", "trade_timestamp"],
        "price": ["price", "trade_price"],
        "size": ["size", "trade_size"],
    }
    found = {}
    lower_map = {str(col).lower(): col for col in columns}
    for key, names in candidates.items():
        for name in names:
            if name.lower() in lower_map:
                found[key] = lower_map[name.lower()]
                break
    missing = [key for key in candidates if key not in found]
    if missing:
        print("Could not auto-detect required trade columns.", flush=True)
        print(list(columns), flush=True)
        raise ValueError(f"Missing required trade logical columns: {missing}")
    print("Detected trade columns:", found, flush=True)
    return found


def parse_timestamp_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.95:
        max_abs = numeric.abs().max()
        if max_abs > 10**17:
            ts = pd.to_datetime(numeric, unit="ns", utc=True, errors="coerce")
        elif max_abs > 10**14:
            ts = pd.to_datetime(numeric, unit="us", utc=True, errors="coerce")
        elif max_abs > 10**11:
            ts = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_convert(EASTERN_TZ)


def parse_option_ticker(df):
    parsed = df["option_ticker"].str.extract(TICKER_PATTERN)
    parsed["expiration"] = pd.to_datetime(parsed["expiration"], format="%y%m%d", errors="coerce")
    parsed["strike"] = pd.to_numeric(parsed["strike"], errors="coerce") / 1000.0
    df["underlying"] = parsed["underlying"]
    df["expiration"] = parsed["expiration"]
    df["option_type"] = parsed["option_type"]
    df["strike"] = parsed["strike"]
    return df


def clean_quotes_chunk(df, trade_date, min_dte=0, max_dte=14):
    if df.empty:
        return df
    df["timestamp"] = parse_timestamp_series(df["raw_timestamp"])
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["bid_size"] = pd.to_numeric(df["bid_size"], errors="coerce")
    df["ask_size"] = pd.to_numeric(df["ask_size"], errors="coerce")

    market_open = pd.Timestamp(f"{trade_date} 09:30:00", tz=EASTERN_TZ)
    market_close = pd.Timestamp(f"{trade_date} 16:00:00", tz=EASTERN_TZ)
    mask = (
        df["timestamp"].notna()
        & df["underlying"].notna()
        & df["expiration"].notna()
        & df["option_type"].isin(["C", "P"])
        & df["strike"].notna()
        & (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["bid"] < df["ask"])
        & (df["timestamp"] >= market_open)
        & (df["timestamp"] <= market_close)
    )
    df = df.loc[mask].copy()
    if df.empty:
        return df

    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    df["relative_spread"] = df["spread"] / df["mid"]
    size_sum = df["ask_size"] + df["bid_size"]
    df["lob_imbalance"] = np.where(size_sum == 0, np.nan, (df["ask_size"] - df["bid_size"]) / size_sum)
    df = df[df["relative_spread"] <= 2.0].copy()
    if df.empty:
        return df

    trade_date_ts = pd.Timestamp(trade_date)
    df["DTE"] = (df["expiration"] - trade_date_ts).dt.days.astype("Int64")
    df = df[(df["DTE"] >= min_dte) & (df["DTE"] <= max_dte)].copy()
    if df.empty:
        return df
    df["option_type_flag"] = (df["option_type"] == "C").astype(int)
    df["date"] = str(trade_date)
    df["timestamp_ns"] = df["timestamp"].dt.tz_convert("UTC").astype("int64")
    return df


def load_trade_events(trade_path, trade_date, universe_symbols, scan_batch_size=2_000_000):
    trade_path = Path(trade_path)
    if not trade_path.exists():
        raise FileNotFoundError(f"Missing trade file: {trade_path}")

    market_open = pd.Timestamp(f"{trade_date} 09:30:00", tz=EASTERN_TZ)
    market_close = pd.Timestamp(f"{trade_date} 16:00:00", tz=EASTERN_TZ)
    kept = []

    def process_chunk(chunk):
        cols = detect_trade_columns(chunk.columns)
        sub = chunk.rename(
            columns={
                cols["ticker"]: "option_ticker",
                cols["timestamp"]: "raw_timestamp",
                cols["price"]: "trade_price",
                cols["size"]: "trade_size",
            }
        ).copy()
        sub["option_ticker"] = sub["option_ticker"].astype("string")
        sub = parse_option_ticker(sub)
        sub = sub[sub["underlying"].isin(universe_symbols)].copy()
        if sub.empty:
            return None
        sub["timestamp"] = parse_timestamp_series(sub["raw_timestamp"])
        sub["trade_price"] = pd.to_numeric(sub["trade_price"], errors="coerce")
        sub["trade_size"] = pd.to_numeric(sub["trade_size"], errors="coerce")
        sub = sub[
            sub["timestamp"].notna()
            & sub["option_ticker"].notna()
            & (sub["trade_price"] > 0)
            & (sub["timestamp"] >= market_open)
            & (sub["timestamp"] <= market_close)
        ].copy()
        if sub.empty:
            return None
        sub["timestamp_ns"] = sub["timestamp"].dt.tz_convert("UTC").astype("int64")
        return sub[["option_ticker", "timestamp_ns", "trade_price", "trade_size"]]

    if trade_path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(trade_path)
        out = process_chunk(raw)
        if out is not None and not out.empty:
            kept.append(out)
    else:
        for chunk in pd.read_csv(trade_path, chunksize=scan_batch_size, low_memory=False):
            out = process_chunk(chunk)
            if out is not None and not out.empty:
                kept.append(out)

    if not kept:
        print(f"No usable trades kept from {trade_path}", flush=True)
        return pd.DataFrame(columns=["option_ticker", "timestamp_ns", "trade_price", "trade_size"])

    out = pd.concat(kept, ignore_index=True)
    print(f"Loaded trade events rows={len(out):,} from {trade_path}", flush=True)
    return out.sort_values(["option_ticker", "timestamp_ns"]).reset_index(drop=True)


def chunk_list(items, batch_size):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def get_unique_underlyings(raw_quote_path, sp100_symbols, scan_batch_size):
    parquet_file = pq.ParquetFile(raw_quote_path)
    columns = detect_columns(parquet_file.schema.names)
    ticker_col = columns["ticker"]
    found = set()
    raw_rows = 0
    for batch in parquet_file.iter_batches(batch_size=scan_batch_size, columns=[ticker_col]):
        chunk = batch.to_pandas()
        raw_rows += len(chunk)
        tickers = chunk[ticker_col].astype("string")
        parsed = tickers.str.extract(TICKER_PATTERN)
        vals = set(parsed["underlying"].dropna().astype(str).tolist())
        found.update(v for v in vals if v in sp100_symbols)
        print(f"Universe scan rows: {raw_rows:,} | unique underlyings so far: {len(found):,}", flush=True)
    underlyings = sorted(found)
    print(f"Final underlying universe size: {len(underlyings):,}", flush=True)
    return underlyings


def write_cleaned_chunk_parquets(raw_quote_path, trade_date, sp100_symbols, underlying_batch_size, scan_batch_size, out_dir, min_dte=0, max_dte=14):
    parquet_file = pq.ParquetFile(raw_quote_path)
    columns = detect_columns(parquet_file.schema.names)
    usecols = [
        columns["ticker"],
        columns["timestamp"],
        columns["bid"],
        columns["ask"],
        columns["bid_size"],
        columns["ask_size"],
    ]
    rename_map = {
        columns["ticker"]: "option_ticker",
        columns["timestamp"]: "raw_timestamp",
        columns["bid"]: "bid",
        columns["ask"]: "ask",
        columns["bid_size"]: "bid_size",
        columns["ask_size"]: "ask_size",
    }

    if len(sp100_symbols) == 1:
        underlyings = sorted(sp100_symbols)
        print(f"Skipping universe scan; single-symbol universe detected: {underlyings}", flush=True)
    else:
        underlyings = get_unique_underlyings(raw_quote_path, sp100_symbols, scan_batch_size)
    batches = chunk_list(underlyings, underlying_batch_size)
    print(f"Processing {len(batches)} underlying batches with size={underlying_batch_size}", flush=True)

    mapping = {}
    for idx, batch in enumerate(batches):
        for sym in batch:
            mapping[sym] = idx

    clean_dir = out_dir / "clean_chunks"
    clean_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = [clean_dir / f"clean_chunk_{idx:03d}.parquet" for idx in range(len(batches))]
    writers = {}
    raw_rows = 0
    kept_rows = 0

    for batch in parquet_file.iter_batches(batch_size=scan_batch_size, columns=usecols):
        chunk = batch.to_pandas()
        raw_rows += len(chunk)
        chunk = chunk.rename(columns=rename_map)
        chunk["option_ticker"] = chunk["option_ticker"].astype("string")
        chunk = parse_option_ticker(chunk)
        chunk = chunk[chunk["underlying"].isin(sp100_symbols)].copy()
        chunk = clean_quotes_chunk(chunk, trade_date, min_dte=min_dte, max_dte=max_dte)
        if chunk.empty:
            print(f"Shard scan rows: {raw_rows:,} | kept cleaned rows so far: {kept_rows:,}", flush=True)
            continue
        chunk["chunk_id"] = chunk["underlying"].map(mapping)
        chunk = chunk.dropna(subset=["chunk_id"]).copy()
        if chunk.empty:
            print(f"Shard scan rows: {raw_rows:,} | kept cleaned rows so far: {kept_rows:,}", flush=True)
            continue
        chunk["chunk_id"] = chunk["chunk_id"].astype(int)
        kept_rows += len(chunk)
        for chunk_id, sub in chunk.groupby("chunk_id", sort=False):
            sub = sub.drop(columns=["chunk_id"]).copy()
            table = pa.Table.from_pandas(sub, preserve_index=False)
            if chunk_id not in writers:
                writers[chunk_id] = pq.ParquetWriter(chunk_paths[chunk_id], table.schema, compression="zstd")
            writers[chunk_id].write_table(table)
        print(f"Shard scan rows: {raw_rows:,} | kept cleaned rows so far: {kept_rows:,}", flush=True)

    for writer in writers.values():
        writer.close()

    existing = [path for path in chunk_paths if path.exists()]
    print(f"Wrote {len(existing)} cleaned chunk parquet files", flush=True)
    return existing
