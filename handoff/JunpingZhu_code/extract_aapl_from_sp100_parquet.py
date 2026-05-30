import argparse
from pathlib import Path

import duckdb


def parse_args():
    parser = argparse.ArgumentParser(description="Extract AAPL rows from an SP100 parquet file.")
    parser.add_argument("--input", required=True, help="Input SP100 parquet path")
    parser.add_argument("--output", required=True, help="Output AAPL parquet path")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"RUN {inp} -> {out}", flush=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    inp_sql = str(inp).replace("'", "''")
    out_sql = str(out).replace("'", "''")
    con.execute(
        f"""
        COPY (
          SELECT *
          FROM read_parquet('{inp_sql}')
          WHERE ticker LIKE 'O:AAPL%'
        )
        TO '{out_sql}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
        """,
    )
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_sql}')").fetchone()[0]
    print(f"DONE rows={n}", flush=True)


if __name__ == "__main__":
    main()
