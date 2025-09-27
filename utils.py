import pandas as pd
import datetime
import numpy as np


def remove_spikes_mad(
    df: pd.DataFrame,
    *,
    price_col: str = "lastSaleYield",  # or "lastSalePrice"
    volume_col: str = "reportedTradeVolume",
    group_keys=("cusip",),  # e.g. ("cusip","issueSymbolIdentifier")
    date_col: str = "tradeDate",
    time_col: str = "tradeTime",
    bucket: str = "1min",  # same cadence you’ll VWAP on
    mad_k: float = 5.0,  # aggressiveness (4–6 typical)
    min_trades: int = 3,  # don’t trim tiny buckets
    q_low: float = 0.001,
    q_high: float = 0.999,  # global clamp fallback
) -> pd.DataFrame:
    s = df.copy()

    s["_ts"] = pd.to_datetime(s[date_col].astype(str) + " " + s[time_col].astype(str), errors="coerce")
    s["_ts"] = s["_ts"].dt.tz_localize("America/New_York")
    s = s.dropna(subset=["_ts", price_col, volume_col]).copy()
    s[price_col] = pd.to_numeric(s[price_col], errors="coerce")
    s[volume_col] = pd.to_numeric(s[volume_col], errors="coerce")
    s = s.dropna(subset=[price_col, volume_col])
    s = s[s[volume_col] > 0]

    lo, hi = s[price_col].quantile([q_low, q_high])
    s = s[(s[price_col] >= lo) & (s[price_col] <= hi)]

    s["_bucket"] = s["_ts"].dt.floor(bucket)

    def _trim_bucket(g: pd.DataFrame) -> pd.DataFrame:
        x = g[price_col].to_numpy(dtype=float)
        if len(x) < min_trades:
            return g
        med = np.median(x)
        dev = np.abs(x - med)
        mad = np.median(dev)
        if mad == 0:  # fallback to IQR if all equal
            q1, q3 = np.quantile(x, [0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                return g
            mask = (x >= q1 - 3 * iqr) & (x <= q3 + 3 * iqr)
        else:
            # 1.4826 scales MAD to sigma-equivalent for normal data
            thresh = mad_k * 1.4826 * mad
            mask = dev <= thresh
        return g.loc[mask]

    cleaned = s.groupby(list(group_keys) + ["_bucket"], group_keys=False, sort=False).apply(_trim_bucket).reset_index(drop=True)

    open = datetime.time(8, 0, 0)
    close = datetime.time(17, 0, 0)
    return cleaned[(cleaned["_ts"].dt.time >= open) & (cleaned["_ts"].dt.time <= close)]


def compute_vwap(
    df: pd.DataFrame,
    interval: str = "1min",
    group_keys=("cusip",),  # or ("cusip","issueSymbolIdentifier"), etc.
    price_col: str = "lastSalePrice",  # pass "lastSaleYield" to VWAY (yield)
    volume_col: str = "reportedTradeVolume",
    date_col: str = "tradeDate",
    time_col: str = "tradeTime",
    add_ohlc: bool = True,
) -> pd.DataFrame:
    needed = set(group_keys) | {price_col, volume_col, date_col, time_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    s = df.copy()
    s["_ts"] = pd.to_datetime(s[date_col].astype(str) + " " + s[time_col].astype(str), errors="coerce")
    s["_ts"] = s["_ts"].dt.tz_localize("America/New_York")

    s = s.dropna(subset=["_ts", price_col, volume_col]).copy()
    s[price_col] = pd.to_numeric(s[price_col], errors="coerce")
    s[volume_col] = pd.to_numeric(s[volume_col], errors="coerce")
    s = s.dropna(subset=[price_col, volume_col])
    s = s[s[volume_col] > 0]

    # PV aggregation
    s["_pv"] = s[price_col] * s[volume_col]
    grp = s.groupby(list(group_keys) + [pd.Grouper(key="_ts", freq=interval, label="left", closed="left")], dropna=False)

    agg = {
        "_pv": ("_pv", "sum"),
        "volume": (volume_col, "sum"),
        "trades": (price_col, "size"),
    }
    if add_ohlc:
        agg.update(
            {
                "open": (price_col, "first"),
                "high": (price_col, "max"),
                "low": (price_col, "min"),
                "close": (price_col, "last"),
            }
        )

    out = grp.agg(**agg)
    out["vwap"] = out["_pv"] / out["volume"]
    out = out.drop(columns=["_pv"])
    out.index = out.index.set_names(list(group_keys) + [f"{interval}_start"])
    return out.sort_index()
