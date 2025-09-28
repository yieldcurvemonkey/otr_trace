import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from FinraFetcher import FinraDataFetcher


def _df_to_parquet(df: pd.DataFrame, path: Path, *, compression: Optional[str] = "zstd"):
    out = df.copy()
    for col in out.columns:
        s = out[col]

    if s.dtype == "object" and s.map(type).nunique() > 1:
        out[col] = s.astype(str)

    path.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(tbl, path, compression=compression)


PATH_STORE = {
    "2Y": Path("./TRACE_2Y"),
    "3Y": Path("./TRACE_3Y"),
    "5Y": Path("./TRACE_5Y"),
    "7Y": Path("./TRACE_7Y"),
    "10Y": Path("./TRACE_10Y"),
    "20Y": Path("./TRACE_20Y"),
    "30Y": Path("./TRACE_30Y"),
}

if __name__ == "__main__":
    ff = FinraDataFetcher(
        debug_verbose=True,
        info_verbose=True,
        error_verbose=True,
    )
    start = datetime.datetime(2025, 9, 2)
    end = datetime.datetime(2025, 9, 5)
    bdays = pd.date_range(start=start, end=end, freq="1b")

    for bd in bdays:
        try:
            benchmarks = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
            intraday_dict = ff.fetch_historcal_trace_trade_history_by_cusip_v2(
                start_date=bd,
                end_date=bd,
                benchmark_terms=benchmarks,
            )
            for bm in benchmarks:
                try:
                    PATH_STORE[bm].mkdir(parents=True, exist_ok=True)
                    output_path = PATH_STORE[bm] / f"{bd.strftime('%Y-%m-%d')}.parquet"
                    _df_to_parquet(intraday_dict[bm], output_path)
                except Exception as e:
                    print(f"error during write for {bm} on {bd}: {e}")
        except Exception as e:
            print(f"error during fetching for {bd}: {e}")
