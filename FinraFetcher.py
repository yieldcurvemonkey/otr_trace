import asyncio
import logging
import math
import random
import time
import warnings
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import aiohttp
import httpx
import pandas as pd
import requests
import ujson as json
from aiohttp.client_exceptions import ClientOSError, ServerDisconnectedError

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


def cookie_string_to_dict(cookie_string):
    cookie_pairs = cookie_string.split("; ")
    cookie_dict = {pair.split("=")[0]: pair.split("=")[1] for pair in cookie_pairs if "=" in pair}
    return cookie_dict


class BaseFetcher:
    def __init__(
        self,
        global_timeout: int = 10,
        proxies: Optional[Dict[str, str]] = None,
        debug_verbose: bool = False,
        info_verbose: bool = False,
        warning_verbose: bool = False,
        error_verbose: bool = False,
    ):
        self._global_timeout = global_timeout
        self._proxies = proxies if proxies else {"http": None, "https": None}
        self._httpx_proxies = {
            "http://": httpx.AsyncHTTPTransport(proxy=self._proxies["http"]),
            "https://": httpx.AsyncHTTPTransport(proxy=self._proxies["https"]),
        }

        self._debug_verbose = debug_verbose
        self._info_verbose = info_verbose
        self._error_verbose = error_verbose
        self._warning_verbose = warning_verbose
        self._setup_logger()

    def _setup_logger(self):
        self._logger = logging.getLogger(self.__class__.__name__)

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._logger.addHandler(handler)

        if self._debug_verbose:
            self._logger.setLevel(logging.DEBUG)
        elif self._info_verbose:
            self._logger.setLevel(logging.INFO)
        elif self._error_verbose:
            self._logger.setLevel(logging.ERROR)
        elif self._warning_verbose:
            self._logger.setLevel(logging.WARNING)
        else:
            self._logger.disabled = True


FINRA_BASE = "https://services-dynarep.ddwa.finra.org"
COOKIE_URL = f"{FINRA_BASE}/public/reporting/v2/group/Firm/name/ActiveIndividual/dynamiclookup/examCode"
DATA_URL = f"{FINRA_BASE}/public/reporting/v2/data/group/FixedIncomeMarket/name/TreasuryTradeHistory"

COMMON_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "application/json",
    "origin": "https://www.finra.org",
    "pragma": "no-cache",
    "referer": "https://www.finra.org/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
}


def build_payload(
    *,
    start_date: datetime,
    end_date: datetime,
    limit: int,
    offset: int,
    cusip: Optional[str] = None,
    benchmark_term: Optional[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]] = None,
) -> Dict:
    if not (cusip or benchmark_term):
        raise ValueError("Need cusip or benchmark_term")
    compare = (
        [{"fieldName": "cusip", "fieldValue": cusip, "compareType": "EQUAL"}]
        if cusip
        else [{"fieldName": "benchmarkTermCode", "fieldValue": benchmark_term, "compareType": "EQUAL"}]
    )
    return {
        "fields": [
            "issueSymbolIdentifier",
            "cusip",
            "tradeDate",
            "tradeTime",
            "reportedTradeVolume",
            "priceType",
            "lastSalePrice",
            "lastSaleYield",
            "reportingSideCode",
            "contraPartyTypeCode",
        ],
        "dateRangeFilters": [
            {
                "fieldName": "tradeDate",
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
            }
        ],
        "compareFilters": compare,
        "limit": limit,
        "offset": offset,
    }


@asynccontextmanager
async def finra_session(
    *, proxies: Optional[Dict[str, str]] = None, max_connections: int = 6, force_close: bool = False, connect_timeout: int = 30, read_timeout: int = 120
):
    timeout = aiohttp.ClientTimeout(total=None, connect=connect_timeout, sock_connect=connect_timeout, sock_read=read_timeout)
    connector = aiohttp.TCPConnector(
        limit=max_connections,
        limit_per_host=max_connections,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        force_close=force_close,
    )
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
        # Warm cookies/XSRF in this same session
        async with session.get(COOKIE_URL, headers=COMMON_HEADERS, proxy=(proxies or {}).get("https")) as r:
            r.raise_for_status()
        yield session


async def _xsrf_token(session: aiohttp.ClientSession) -> Optional[str]:
    jar = session.cookie_jar.filter_cookies(FINRA_BASE)
    xsrf = jar.get("XSRF-TOKEN")
    return xsrf.value if xsrf else None


async def post_with_retry(
    session: aiohttp.ClientSession, *, json_body: Dict, proxies: Optional[Dict[str, str]] = None, max_retries: int = 5, base_delay: float = 0.5
):
    headers = dict(COMMON_HEADERS)
    token = await _xsrf_token(session)
    if token:
        headers["x-xsrf-token"] = token

    for attempt in range(max_retries):
        try:
            async with session.post(
                DATA_URL,
                headers=headers,
                json=json_body,
                proxy=(proxies or {}).get("https"),
            ) as resp:
                # Handle auth/overload with backoff and cookie refresh
                if resp.status in (401, 403, 429, 500, 502, 503, 504):
                    if resp.status in (401, 403):
                        try:
                            async with session.get(COOKIE_URL, headers=COMMON_HEADERS, proxy=(proxies or {}).get("https")) as r2:
                                r2.raise_for_status()
                        except Exception:
                            pass
                    delay = base_delay * (2**attempt) + random.random() * 0.2
                    await asyncio.sleep(delay)
                    continue

                resp.raise_for_status()
                return await resp.json()

        except (ServerDisconnectedError, ClientOSError, asyncio.TimeoutError):
            delay = base_delay * (2**attempt) + random.random() * 0.2
            await asyncio.sleep(delay)
            # Refresh cookies once after a networkish failure
            try:
                async with session.get(COOKIE_URL, headers=COMMON_HEADERS, proxy=(proxies or {}).get("https")) as r3:
                    r3.raise_for_status()
            except Exception:
                pass

    raise RuntimeError("FINRA POST failed after retries")


async def fetch_one_page(session, *, start_date, end_date, offset, cusip: Optional[str] = None, benchmark_term: Optional[str] = None, proxies=None) -> pd.DataFrame:
    body = build_payload(start_date=start_date, end_date=end_date, limit=5000, offset=offset, cusip=cusip, benchmark_term=benchmark_term)
    res = await post_with_retry(session, json_body=body, proxies=proxies)
    # headers like Record-Total live at res["returnBody"]["headers"], but page data is:
    trade_data_json = json.loads(res["returnBody"]["data"])
    return pd.DataFrame(trade_data_json)


async def probe_total_records_for_key(
    session, *, start_date, end_date, cusip: Optional[str] = None, benchmark_term: Optional[str] = None, proxies=None
) -> Tuple[str, int]:
    body = build_payload(start_date=start_date, end_date=end_date, limit=1, offset=1, cusip=cusip, benchmark_term=benchmark_term)
    res = await post_with_retry(session, json_body=body, proxies=proxies)
    total = int(res["returnBody"]["headers"]["Record-Total"][0])
    key = cusip if cusip else benchmark_term
    return key, total


async def probe_totals(
    session, *, start_date, end_date, cusips: Optional[List[str]] = None, benchmark_terms: Optional[List[str]] = None, proxies=None, max_in_flight: int = 4
) -> Dict[str, int]:
    sem = asyncio.Semaphore(max_in_flight)

    async def _task_c(c):
        async with sem:
            return await probe_total_records_for_key(session, start_date=start_date, end_date=end_date, cusip=c, proxies=proxies)

    async def _task_b(b):
        async with sem:
            return await probe_total_records_for_key(session, start_date=start_date, end_date=end_date, benchmark_term=b, proxies=proxies)

    tasks = []
    if cusips:
        tasks += [_task_c(c) for c in cusips]
    if benchmark_terms:
        tasks += [_task_b(b) for b in benchmark_terms]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: Dict[str, int] = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        k, v = r
        out[k] = v
    return out


async def fetch_all_pages_for_key(
    session, *, start_date, end_date, key_is_cusip: bool, key_value: str, total_records: int, proxies=None, max_in_flight: int = 3
) -> pd.DataFrame:
    sem = asyncio.Semaphore(max_in_flight)
    pages = math.ceil(total_records / 5000)

    async def _task(offset):
        async with sem:
            return await fetch_one_page(
                session,
                start_date=start_date,
                end_date=end_date,
                offset=offset,
                cusip=key_value if key_is_cusip else None,
                benchmark_term=key_value if not key_is_cusip else None,
                proxies=proxies,
            )

    frames = await asyncio.gather(*[_task(i * 5000) for i in range(pages + 1)], return_exceptions=True)
    dfs = [fr for fr in frames if isinstance(fr, pd.DataFrame) and not fr.empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).sort_values(["tradeDate", "tradeTime"]).reset_index(drop=True)


# --- single request helper that returns (headers, rows) ----------------------
async def _fetch_page(session: aiohttp.ClientSession, *, body: dict, proxies=None):
    headers = dict(COMMON_HEADERS)
    token = await _xsrf_token(session)
    if token:
        headers["x-xsrf-token"] = token

    async with session.post(DATA_URL, headers=headers, json=body, proxy=(proxies or {}).get("https")) as resp:
        # (minor) only refresh cookies on 401/403; otherwise backoff/retry outside if needed
        resp.raise_for_status()
        data = await resp.read()  # <- bytes
        obj = json.loads(data)  # <- faster than resp.json()
        rb = obj["returnBody"]
        # rows are embedded as a JSON string; decode once
        rows = json.loads(rb["data"]) if rb["data"] else []
        return rb.get("headers", {}), rows


# --- per-key pipeline: fetch first page, then schedule the rest -------------
async def _fetch_key_all_pages(
    session,
    *,
    start_date,
    end_date,
    key_is_cusip: bool,
    key_value: str,
    proxies=None,
    max_in_flight: int = 6,
    build_df: bool = True,
    sort_final: bool = False,
):
    # 1) first page doubles as "probe"
    first_body = build_payload(
        start_date=start_date,
        end_date=end_date,
        limit=5000,
        offset=0,
        cusip=key_value if key_is_cusip else None,
        benchmark_term=key_value if not key_is_cusip else None,
    )
    hdrs, first_rows = await _fetch_page(session, body=first_body, proxies=proxies)
    total = int(hdrs.get("Record-Total", [0])[0]) if hdrs else 0
    if total <= 0:
        return key_value, (pd.DataFrame() if build_df else [])

    # 2) schedule remaining pages immediately (pipeline)
    pages = math.ceil(total / 5000)
    # offsets for remaining pages only
    offsets = [i * 5000 for i in range(1, pages)]

    sem = asyncio.Semaphore(max_in_flight)

    async def _task(off):
        async with sem:
            body = build_payload(
                start_date=start_date,
                end_date=end_date,
                limit=5000,
                offset=off,
                cusip=key_value if key_is_cusip else None,
                benchmark_term=key_value if not key_is_cusip else None,
            )
            _, rows = await _fetch_page(session, body=body, proxies=proxies)
            return rows

    # Important: stream results as they complete; don’t wait for all
    tasks = [asyncio.create_task(_task(off)) for off in offsets]
    all_rows = list(first_rows)
    for t in asyncio.as_completed(tasks):
        rows = await t
        if rows:
            all_rows.extend(rows)

    if not build_df:
        return key_value, all_rows

    # 3) build DataFrame off the event loop
    loop = asyncio.get_running_loop()

    def _to_df():
        df = pd.DataFrame.from_records(all_rows)
        # optional cheap dtype normalization
        try:
            df["reportedTradeVolume"] = df["reportedTradeVolume"].astype("Int64")
        except Exception:
            pass
        for col in ("lastSalePrice", "lastSaleYield"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if sort_final and {"tradeDate", "tradeTime"} <= set(df.columns):
            # Sorting is expensive; keep it optional
            df = df.sort_values(["tradeDate", "tradeTime"], kind="mergesort")
        return df.reset_index(drop=True)

    df = await loop.run_in_executor(None, _to_df)
    return key_value, df


class FinraDataFetcher(BaseFetcher):
    def __init__(
        self,
        global_timeout: int = 10,
        proxies: Optional[Dict[str, str]] = None,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        super().__init__(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

    def fetch_historcal_trace_trade_history_by_cusip_v3(
        self,
        start_date: datetime,
        end_date: datetime,
        cusips: Optional[List[str]] = None,
        benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]]] = None,
        xlsx_path: Optional[str] = None,
        session_timeout_minutes: int = 5,
        max_connections: int = 10,
        max_in_flight: int = 6,
        sort_final: bool = False,
    ):
        assert start_date == end_date, "only 1 day of data supported!"
        t0 = time.time()

        async def _run():
            async with finra_session(
                proxies=self._proxies,
                max_connections=max_connections,
                connect_timeout=session_timeout_minutes * 60,
                read_timeout=session_timeout_minutes * 60,
            ) as session:
                coros = []
                if cusips:
                    for c in cusips:
                        coros.append(
                            _fetch_key_all_pages(
                                session,
                                start_date=start_date,
                                end_date=end_date,
                                key_is_cusip=True,
                                key_value=c,
                                proxies=self._proxies,
                                max_in_flight=max_in_flight,
                                sort_final=sort_final,
                            )
                        )
                if benchmark_terms:
                    for b in benchmark_terms:
                        coros.append(
                            _fetch_key_all_pages(
                                session,
                                start_date=start_date,
                                end_date=end_date,
                                key_is_cusip=False,
                                key_value=b,
                                proxies=self._proxies,
                                max_in_flight=max_in_flight,
                                sort_final=sort_final,
                            )
                        )

                results = await asyncio.gather(*coros, return_exceptions=True)

                out: Dict[str, pd.DataFrame] = {}
                for r in results:
                    if isinstance(r, Exception):
                        continue
                    k, df = r
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        out[k] = df

                # optional Excel write (slow); prefer Parquet in production
                if xlsx_path and out:
                    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
                        for k, df in out.items():
                            df.to_excel(w, sheet_name=str(k)[:31], index=False)
                return out

        res = asyncio.run(_run())
        self._logger.info(f"TRACE v3 - Total: {time.time()-t0:.3f}s, keys={len(res)}")
        return res

    def fetch_historcal_trace_trade_history_by_cusip_v2(
        self,
        start_date: datetime,
        end_date: datetime,
        cusips: Optional[List[str]] = None,
        benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]]] = None,
        xlsx_path: Optional[str] = None,
        session_timeout_minutes: Optional[int] = 5,
        max_connections: int = 6,
        max_in_flight: int = 3,
        force_close: bool = False,  # set True if you keep seeing disconnects
    ):
        assert start_date == end_date, "only 1 day of data supported!"
        total_t1 = time.time()

        async def _run():
            # one session for cookie warmup + probes + page fetches
            async with finra_session(
                proxies=self._proxies,
                max_connections=max_connections,
                force_close=force_close,
                connect_timeout=session_timeout_minutes * 60,
                read_timeout=session_timeout_minutes * 60,
            ) as session:

                # 1) probe totals (NO asyncio.run() here)
                t1 = time.time()
                totals = await probe_totals(
                    session,
                    start_date=start_date,
                    end_date=end_date,
                    cusips=cusips,
                    benchmark_terms=benchmark_terms,
                    proxies=self._proxies,
                    max_in_flight=max_in_flight,
                )
                self._logger.info(f"TRACE - FINRA totals probe took: {time.time() - t1:.3f}s")
                self._logger.debug(f"TRACE - totals: {totals}")

                # 2) fetch pages per key in waves
                fetch_tasks = []
                if cusips:
                    for c in cusips:
                        total = int(totals.get(c, 0))
                        if total <= 0:
                            self._logger.debug(f"TRACE - {c} has no records or probe failed.")
                            continue
                        fetch_tasks.append(
                            fetch_all_pages_for_key(
                                session,
                                start_date=start_date,
                                end_date=end_date,
                                key_is_cusip=True,
                                key_value=c,
                                total_records=total,
                                proxies=self._proxies,
                                max_in_flight=max_in_flight,
                            )
                        )
                elif benchmark_terms:
                    for b in benchmark_terms:
                        total = int(totals.get(b, 0))
                        if total <= 0:
                            self._logger.debug(f"TRACE - {b} has no records or probe failed.")
                            continue
                        fetch_tasks.append(
                            fetch_all_pages_for_key(
                                session,
                                start_date=start_date,
                                end_date=end_date,
                                key_is_cusip=False,
                                key_value=b,
                                total_records=total,
                                proxies=self._proxies,
                                max_in_flight=max_in_flight,
                            )
                        )
                else:
                    raise ValueError("Provide 'cusips' or 'benchmark_terms'.")

                fetch_all_t1 = time.time()
                frames = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                self._logger.info(f"TRACE - Fetch All Took: {time.time() - fetch_all_t1:.3f}s")

                # Build dict keyed by cusip/term in same order we queued
                concatenated_dfs: Dict[str, pd.DataFrame] = {}
                idx = 0
                if cusips:
                    for c in cusips:
                        total = int(totals.get(c, 0))
                        if total <= 0:
                            continue
                        fr = frames[idx]
                        idx += 1
                        if isinstance(fr, Exception) or fr is None or fr.empty:
                            continue
                        concatenated_dfs[c] = fr
                else:
                    for b in benchmark_terms or []:
                        total = int(totals.get(b, 0))
                        if total <= 0:
                            continue
                        fr = frames[idx]
                        idx += 1
                        if isinstance(fr, Exception) or fr is None or fr.empty:
                            continue
                        concatenated_dfs[b] = fr

                # 3) optional Excel write
                if xlsx_path and concatenated_dfs:
                    t1 = time.time()
                    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                        for key, df in concatenated_dfs.items():
                            df.to_excel(writer, sheet_name=str(key)[:31], index=False)
                    self._logger.info(f"TRACE - XLSX Write Took: {time.time() - t1:.3f}s")

                return concatenated_dfs

        # NOTE: top-level only — do not call asyncio.run() inside other coroutines
        results = asyncio.run(_run())
        self._logger.info(f"TRACE - Total Time Elapsed: {time.time() - total_t1:.3f}s")
        return results

    def fetch_historcal_trace_trade_history_by_cusip(
        self,
        start_date: datetime,
        end_date: datetime,
        cusips: Optional[List[str]] = None,
        benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]]] = None,
        xlsx_path: Optional[str] = None,
        session_timeout_minutes: Optional[int] = 5,
    ):
        assert start_date == end_date, "only 1 day of data supported!"
        total_t1 = time.time()

        async def build_fetch_tasks_historical_trace_data(
            session: aiohttp.ClientSession,
            start_date: datetime,
            end_date: datetime,
            cusips: Optional[List[str]] = None,
            benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]]] = None,
            uid: Optional[str | int] = None,
        ):
            finra_cookie_headers = {
                "authority": "services-dynarep.ddwa.finra.org",
                "method": "OPTIONS",
                "path": "/public/reporting/v2/data/group/FixedIncomeMarket/name/TreasuryTradeHistory",
                "scheme": "https",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br, zstd",
                "accept-language": "en-US,en;q=0.9",
                "access-control-request-headers": "content-type,x-xsrf-token",
                "access-control-request-method": "POST",
                "cache-control": "no-cache",
                "origin": "https://www.finra.org",
                "pragma": "no-cache",
                "priority": "u=1, i",
                "referer": "https://www.finra.org/",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-site",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            }

            finra_cookie_t1 = time.time()
            finra_cookie_url = "https://services-dynarep.ddwa.finra.org/public/reporting/v2/group/Firm/name/ActiveIndividual/dynamiclookup/examCode"
            finra_cookie_response = requests.get(finra_cookie_url, headers=finra_cookie_headers, proxies=self._proxies)
            if not finra_cookie_response.ok:
                raise ValueError(f"TRACE - FINRA Cookies Request Bad Status: {finra_cookie_response.status_code}")
            finra_cookie_str = dict(finra_cookie_response.headers)["Set-Cookie"]
            finra_cookie_dict = cookie_string_to_dict(cookie_string=finra_cookie_str)
            self._logger.info(f"TRACE - FINRA Cookie Fetch Took: {time.time() - finra_cookie_t1} seconds")

            def build_finra_trade_history_headers(cookie_str: str, x_xsrf_token_str: str):
                return {
                    "authority": "services-dynarep.ddwa.finra.org",
                    "method": "POST",
                    "path": "/public/reporting/v2/data/group/FixedIncomeMarket/name/TreasuryTradeHistory",
                    "scheme": "https",
                    "accept": "application/json, text/plain, */*",
                    "accept-encoding": "gzip, deflate, br, zstd",
                    "accept-language": "en-US,en;q=0.9",
                    "cache-control": "no-cache",
                    "content-type": "application/json",
                    "dnt": "1",
                    "origin": "https://www.finra.org",
                    "pragma": "no-cache",
                    "priority": "u=1, i",
                    "referer": "https://www.finra.org/",
                    "sec-ch-ua": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"Windows"',
                    "sec-fetch-dest": "empty",
                    "sec-fetch-mode": "cors",
                    "sec-fetch-site": "same-site",
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
                    "x-xsrf-token": x_xsrf_token_str,
                    "cookie": cookie_str,
                }

            # maps size of trade history records between given start and end dates of said cusip
            def build_finra_trade_history_payload(
                start_date: datetime,
                end_date: datetime,
                limit: int,
                offset: int,
                cusip: Optional[str] = None,
                benchmark_term: Optional[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]] = None,
            ) -> Dict[str, int]:
                if cusip:
                    filter_payload = [
                        {
                            "fieldName": "cusip",
                            "fieldValue": cusip,
                            "compareType": "EQUAL",
                        }
                    ]
                elif benchmark_term:
                    filter_payload = [
                        {
                            "fieldName": "benchmarkTermCode",
                            "fieldValue": benchmark_term,
                            "compareType": "EQUAL",
                        }
                    ]
                else:
                    raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (build_finra_trade_history_payload)")
                return {
                    "fields": [
                        "issueSymbolIdentifier",
                        "cusip",
                        "tradeDate",
                        "tradeTime",
                        "reportedTradeVolume",
                        "priceType",
                        "lastSalePrice",
                        "lastSaleYield",
                        "reportingSideCode",
                        "contraPartyTypeCode",
                    ],
                    "dateRangeFilters": [
                        {
                            "fieldName": "tradeDate",
                            "startDate": start_date.strftime("%Y-%m-%d"),
                            "endDate": end_date.strftime("%Y-%m-%d"),
                        },
                    ],
                    "compareFilters": filter_payload,
                    "limit": limit,  # 5000 is Max Limit
                    "offset": offset,
                }

            def get_cusips_finra_pagination_configs(
                start_date: datetime,
                end_date: datetime,
                cusips: Optional[List[str]] = None,
                benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]]] = None,
            ):
                async def fetch_finra_cusip_pagination_config(
                    config_session: aiohttp.ClientSession,
                    start_date: datetime,
                    end_date: datetime,
                    cusip: Optional[str] = None,
                    benchmark_term: Optional[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]] = None,
                ):
                    try:
                        url = "https://services-dynarep.ddwa.finra.org/public/reporting/v2/data/group/FixedIncomeMarket/name/TreasuryTradeHistory"
                        config_response = await config_session.post(
                            url,
                            headers=build_finra_trade_history_headers(
                                cookie_str=finra_cookie_str,
                                x_xsrf_token_str=finra_cookie_dict["XSRF-TOKEN"],
                            ),
                            json=build_finra_trade_history_payload(
                                start_date=start_date, end_date=end_date, limit=1, offset=1, cusip=cusip, benchmark_term=benchmark_term
                            ),
                            proxy=self._proxies["https"],
                        )
                        config_response.raise_for_status()
                        record_total_json = await config_response.json()
                        record_total_str = record_total_json["returnBody"]["headers"]["Record-Total"][0]

                        if cusip:
                            return cusip, record_total_str
                        elif benchmark_term:
                            return benchmark_term, record_total_str
                        else:
                            raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (get_cusips_finra_pagination_configs)")

                    except aiohttp.ClientResponseError:
                        self._logger.error(f"TRACE - CONFIGs Bad Status: {config_response.status}")
                        record_total_str = -1
                        if cusip:
                            return cusip, record_total_str
                        elif benchmark_term:
                            return benchmark_term, record_total_str
                        else:
                            raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (get_cusips_finra_pagination_configs)")

                    except Exception as e:
                        self._logger.error(f"TRACE - CONFIGs Error : {str(e)}")
                        record_total_str = -1
                        if cusip:
                            return cusip, record_total_str
                        elif benchmark_term:
                            return benchmark_term, record_total_str
                        else:
                            raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (get_cusips_finra_pagination_configs)")

                async def build_finra_config_tasks(
                    config_session: aiohttp.ClientSession,
                    start_date: datetime,
                    end_date: datetime,
                    cusips: Optional[List[str]] = None,
                    benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]]] = None,
                ):
                    if cusips:
                        tasks = [
                            fetch_finra_cusip_pagination_config(
                                config_session=config_session,
                                start_date=start_date,
                                end_date=end_date,
                                cusip=cusip,
                            )
                            for cusip in cusips
                        ]
                    elif benchmark_terms:
                        tasks = [
                            fetch_finra_cusip_pagination_config(
                                config_session=config_session, start_date=start_date, end_date=end_date, benchmark_term=benchmark_term
                            )
                            for benchmark_term in benchmark_terms
                        ]
                    else:
                        raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (build_finra_config_tasks)")

                    return await asyncio.gather(*tasks)

                async def run_fetch_all(
                    start_date: datetime,
                    end_date: datetime,
                    cusips: Optional[List[str]] = None,
                    benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]]] = None,
                ) -> List[pd.DataFrame]:
                    async with aiohttp.ClientSession() as config_session:
                        all_data = await build_finra_config_tasks(
                            config_session=config_session, start_date=start_date, end_date=end_date, cusips=cusips, benchmark_terms=benchmark_terms
                        )
                        return all_data

                cusip_finra_api_payload_configs = dict(
                    asyncio.run(
                        run_fetch_all(
                            start_date=start_date,
                            end_date=end_date,
                            cusips=cusips,
                            benchmark_terms=benchmark_terms,
                        )
                    )
                )
                return cusip_finra_api_payload_configs

            cusip_finra_api_payload_configs_t1 = time.time()
            cusip_finra_api_payload_configs = get_cusips_finra_pagination_configs(
                start_date=start_date, end_date=end_date, cusips=cusips, benchmark_terms=benchmark_terms
            )
            self._logger.info(f"TRACE - FINRA CUSIP API Payload Configs Took: {time.time() - cusip_finra_api_payload_configs_t1} seconds")
            self._logger.debug(f"TRACE - CUSIP API Payload Configs: {cusip_finra_api_payload_configs}")

            async def fetch_finra_cusip_trade_history(
                session: aiohttp.ClientSession,
                start_date: datetime,
                end_date: datetime,
                offset: int,
                cusip: Optional[str] = None,
                benchmark_term: Optional[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]] = None,
            ):
                try:
                    url = "https://services-dynarep.ddwa.finra.org/public/reporting/v2/data/group/FixedIncomeMarket/name/TreasuryTradeHistory"
                    response = await session.post(
                        url,
                        headers=build_finra_trade_history_headers(
                            cookie_str=finra_cookie_str,
                            x_xsrf_token_str=finra_cookie_dict["XSRF-TOKEN"],
                        ),
                        json=build_finra_trade_history_payload(
                            start_date=start_date, end_date=end_date, limit=5000, offset=offset, cusip=cusip, benchmark_term=benchmark_term
                        ),
                        proxy=self._proxies["https"],
                    )
                    response.raise_for_status()
                    trade_history_json = await response.json()
                    trade_data_json = json.loads(trade_history_json["returnBody"]["data"])
                    df = pd.DataFrame(trade_data_json)
                    if cusip:
                        if uid:
                            return cusip, df, uid
                        return cusip, df
                    elif benchmark_term:
                        if uid:
                            return benchmark_term, df, uid
                        return benchmark_term, df
                    else:
                        raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (fetch_finra_cusip_trade_history)")

                except aiohttp.ClientResponseError:
                    self._logger.error(f"TRACE - Trade History Bad Status: {response.status}")
                    if cusip:
                        if uid:
                            return cusip, None, uid
                        return cusip, None
                    elif benchmark_term:
                        if uid:
                            return benchmark_term, None, uid
                        return benchmark_term, None
                    else:
                        raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (fetch_finra_cusip_trade_history)")

                except Exception as e:
                    self._logger.error(f"TRACE - Trade History Error : {str(e)}")
                    if cusip:
                        if uid:
                            return cusip, None, uid
                        return cusip, None
                    elif benchmark_term:
                        if uid:
                            return benchmark_term, None, uid
                        return benchmark_term, None
                    else:
                        raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (fetch_finra_cusip_trade_history)")

            tasks = []
            if cusips:
                for cusip in cusips:
                    max_record_size = int(cusip_finra_api_payload_configs[cusip])
                    if max_record_size == -1:
                        self._logger.debug(f"TRACE - {cusip} had -1 Max Record Size - Does it Exist?")
                        continue
                    num_reqs = math.ceil(max_record_size / 5000)
                    self._logger.debug(f"TRACE - {cusip} Reqs: {num_reqs}")
                    offsets = []
                    for i in range(0, num_reqs + 1):
                        curr_offset = i * 5000
                        offsets.append(curr_offset)
                        # failing one on purpose
                        # if curr_offset > max_record_size:
                        #     curr_offset = max_record_size
                        tasks.append(
                            fetch_finra_cusip_trade_history(
                                session=session,
                                cusip=cusip,
                                start_date=start_date,
                                end_date=end_date,
                                offset=curr_offset,
                            )
                        )
            elif benchmark_terms:
                for benchmark_term in benchmark_terms:
                    max_record_size = int(cusip_finra_api_payload_configs[benchmark_term])
                    if max_record_size == -1:
                        self._logger.debug(f"TRACE - {benchmark_term} had -1 Max Record Size - Does it Exist?")
                        continue
                    num_reqs = math.ceil(max_record_size / 5000)
                    self._logger.debug(f"TRACE - {benchmark_term} Reqs: {num_reqs}")
                    offsets = []
                    for i in range(0, num_reqs + 1):
                        curr_offset = i * 5000
                        offsets.append(curr_offset)
                        # failing one on purpose
                        # if curr_offset > max_record_size:
                        # curr_offset = max_record_size
                        tasks.append(
                            fetch_finra_cusip_trade_history(
                                session=session,
                                benchmark_term=benchmark_term,
                                start_date=start_date,
                                end_date=end_date,
                                offset=curr_offset,
                            )
                        )
            else:
                raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (build_fetch_tasks_historical_trace_data)")

            # self._logger.debug(f"LEN OF TASKS: {len(tasks)}")
            # self._logger.debug(f"OFFSETS Used: {offsets}")
            return tasks

        async def build_tasks(
            session: aiohttp.ClientSession,
            start_date: datetime,
            end_date: datetime,
            cusips: Optional[List[str]] = None,
            benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]]] = None,
        ):
            tasks = await build_fetch_tasks_historical_trace_data(
                session=session, start_date=start_date, end_date=end_date, cusips=cusips, benchmark_terms=benchmark_terms
            )
            return await asyncio.gather(*tasks)

        async def run_fetch_all(
            start_date: datetime,
            end_date: datetime,
            cusips: Optional[List[str]] = None,
            benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]]] = None,
        ):
            session_timeout = aiohttp.ClientTimeout(
                total=None,
                sock_connect=session_timeout_minutes * 60,
                sock_read=session_timeout_minutes * 60,
            )
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                all_data = await build_tasks(session=session, start_date=start_date, end_date=end_date, cusips=cusips, benchmark_terms=benchmark_terms)
                return all_data

        fetch_all_t1 = time.time()
        results: List[Tuple[str, pd.DataFrame]] = asyncio.run(
            run_fetch_all(start_date=start_date, end_date=end_date, cusips=cusips, benchmark_terms=benchmark_terms)
        )
        self._logger.info(f"TRACE - Fetch All Took: {time.time() - fetch_all_t1} seconds")
        dfs_by_key = defaultdict(list)
        for key, df in results:
            if df is None:
                continue
            dfs_by_key[key].append(df)

        df_concatation_t1 = time.time()
        concatenated_dfs = {key: pd.concat(dfs).sort_values(by=["tradeDate", "tradeTime"]).reset_index(drop=True) for key, dfs in dfs_by_key.items()}
        self._logger.info(f"TRACE - DF Concation Took: {time.time() - df_concatation_t1} seconds")

        if xlsx_path:
            xlsx_write_t1 = time.time()
            with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                for key, df in concatenated_dfs.items():
                    df.to_excel(writer, sheet_name=key, index=False)

            self._logger.info(f"TRACE - XLSX Write Took: {time.time() - xlsx_write_t1} seconds")

        self._logger.info(f"TRACE - Total Time Elapsed: {time.time() - total_t1} seconds")

        return concatenated_dfs
