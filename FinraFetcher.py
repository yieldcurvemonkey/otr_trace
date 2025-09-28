import asyncio
import itertools
import logging
import math
import random
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import aiohttp
import httpx
import pandas as pd
import requests
import ujson as json
from aiohttp.client_exceptions import ClientOSError, ServerDisconnectedError
from httpx_socks import AsyncProxyTransport

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


NORDVPN_HOSTS = [
    # "amsterdam.nl.socks.nordhold.net",
    "atlanta.us.socks.nordhold.net",  # good
    "chicago.us.socks.nordhold.net",  # good
    "dallas.us.socks.nordhold.net",  # good
    "los-angeles.us.socks.nordhold.net",
    "new-york.us.socks.nordhold.net",
    "phoenix.us.socks.nordhold.net",
    "san-francisco.us.socks.nordhold.net",  # good
    # "stockholm.se.socks.nordhold.net", # good
    # "nl.socks.nordhold.net",
    # "se.socks.nordhold.net", # good
    "us.socks.nordhold.net",  # good
    None,
]
random.shuffle(NORDVPN_HOSTS)
proxy_cycler = itertools.cycle(NORDVPN_HOSTS)


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


def cookie_string_to_dict(cookie_string):
    cookie_pairs = cookie_string.split("; ")
    cookie_dict = {pair.split("=")[0]: pair.split("=")[1] for pair in cookie_pairs if "=" in pair}
    return cookie_dict


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


class FinraDataFetcher(BaseFetcher):
    def __init__(
        self,
        global_timeout: int = 10,
        proxies: Optional[Dict[str, str]] = None,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        super().__init__(global_timeout=global_timeout, proxies=proxies, debug_verbose=debug_verbose, info_verbose=info_verbose, error_verbose=error_verbose)
        self._finra_proxy_state = {"proxy_url": None, "host": None, "chosen_at": 0.0, "ttl": 90}

    @staticmethod
    def _build_proxy_url(host: Optional[str]) -> Optional[str]:
        if not host:
            return None
        import os
        from urllib.parse import quote

        user = os.getenv("NORDVPN_USER") or ""
        pwd = os.getenv("NORDVPN_PASS") or ""
        # socks5h so DNS resolves through the proxy
        return f"socks5h://{quote(user, safe='')}:{quote(pwd, safe='')}@{host}:1080"

    @staticmethod
    def _requests_proxies(proxy_url: Optional[str]) -> Optional[dict]:
        return {"http": proxy_url, "https": proxy_url} if proxy_url else None

    def _preflight_proxy(self, proxy_url: Optional[str], timeout: int = 6) -> bool:
        """Quick external IP check to validate POP. Accepts a *string URL* or None."""
        try:
            r = requests.get(
                "https://api.ipify.org?format=json",
                proxies=self._requests_proxies(proxy_url),
                timeout=timeout,
                headers={"Connection": "close"},
            )
            r.raise_for_status()
            return True
        except Exception:
            return False

    def _choose_proxy_url(self) -> tuple[Optional[str], Optional[str]]:
        """Rotate through NORDVPN_HOSTS; return (proxy_url, host)."""
        for _ in range(len(NORDVPN_HOSTS)):
            host = next(proxy_cycler)
            if host is None:
                return None, None  # DIRECT
            proxy_url = self._build_proxy_url(host)
            if self._preflight_proxy(proxy_url):
                return proxy_url, host
        return None, None  # fall back to DIRECT

    def _get_sticky_proxy_url(self, *, ttl: int = 90, rotate: bool = False) -> Optional[str]:
        S = self._finra_proxy_state
        now = time.time()
        if rotate or (now - S["chosen_at"] >= S["ttl"]):
            proxy_url, host = self._choose_proxy_url()
            S.update({"proxy_url": proxy_url, "host": host, "chosen_at": now, "ttl": ttl})
            self._logger.info(f"TRACE - FINRA proxy set: {host or 'DIRECT'}")
        return S["proxy_url"]

    def fetch_historcal_trace_trade_history_by_cusip_v3(
        self,
        start_date: datetime,
        end_date: datetime,
        cusips: Optional[List[str]] = None,
        benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]]] = None,
        xlsx_path: Optional[str] = None,
        session_timeout_minutes: Optional[int] = 5,
        max_connections: int = 10,
        max_in_flight: int = 6,
    ):
        assert start_date == end_date, "only 1 day of data supported!"
        total_t1 = time.time()

        async def _build_h2_client(proxy_url: Optional[str]) -> "httpx.AsyncClient":
            # Prefer built-in httpx.Proxy mounts (httpx ≥ 0.28). Fallback to httpx_socks.
            limits = httpx.Limits(max_keepalive_connections=max_connections, max_connections=max_connections)
            timeout = httpx.Timeout(connect=session_timeout_minutes * 60, read=session_timeout_minutes * 60, write=session_timeout_minutes * 60, pool=None)
            mounts = {}
            try:
                # Built-in proxy (supports socks5h in modern httpx)
                if proxy_url:
                    proxy = httpx.Proxy(proxy_url)  # type: ignore[attr-defined]
                    mounts = {"all://": proxy}
                    return httpx.AsyncClient(http2=True, headers=COMMON_HEADERS, timeout=timeout, limits=limits, mounts=mounts, trust_env=False)
            except Exception:
                pass

            if proxy_url:
                # Fallback to httpx_socks transport
                try:
                    transport = AsyncProxyTransport.from_url(proxy_url)
                    mounts = {"all://": transport}
                    return httpx.AsyncClient(headers=COMMON_HEADERS, timeout=timeout, mounts=mounts, trust_env=False)
                except Exception as e:
                    self._logger.warning(f"TRACE(h2) - proxy mount failed, falling back DIRECT: {e}")

            # DIRECT with HTTP/2 via mounted transport
            transport = httpx.AsyncHTTPTransport(http2=True, limits=limits)
            mounts = {"all://": transport}
            return httpx.AsyncClient(headers=COMMON_HEADERS, timeout=timeout, mounts=mounts, trust_env=False)

        # ---------- http2 helpers using the mounted client ----------
        async def _xsrf_token(client: httpx.AsyncClient) -> Optional[str]:
            return client.cookies.get("XSRF-TOKEN")

        async def _post_with_retry(client: httpx.AsyncClient, *, json_body: Dict, max_retries: int = 5, base_delay: float = 0.5):
            headers = dict(COMMON_HEADERS)
            token = await _xsrf_token(client)
            if token:
                headers["x-xsrf-token"] = token
            last_exc = None
            for attempt in range(max_retries):
                try:
                    resp = await client.post(DATA_URL, headers=headers, json=json_body)
                    if resp.status_code in (401, 403, 429, 500, 502, 503, 504):
                        if resp.status_code in (401, 403):
                            try:
                                warm = await client.get(COOKIE_URL, headers=COMMON_HEADERS)
                                warm.raise_for_status()
                                headers["x-xsrf-token"] = await _xsrf_token(client) or headers.get("x-xsrf-token")
                            except Exception:
                                pass
                        delay = base_delay * (2**attempt) + random.random() * 0.2
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
                    return json.loads(await resp.aread())
                except Exception as e:
                    last_exc = e
                    delay = base_delay * (2**attempt) + random.random() * 0.2
                    await asyncio.sleep(delay)
                    try:
                        warm = await client.get(COOKIE_URL, headers=COMMON_HEADERS)
                        warm.raise_for_status()
                        headers["x-xsrf-token"] = await _xsrf_token(client) or headers.get("x-xsrf-token")
                    except Exception:
                        pass
            raise RuntimeError(f"FINRA POST failed after retries: {last_exc}") from last_exc

        async def _fetch_one_page(
            client: httpx.AsyncClient, *, start_date, end_date, offset, cusip: Optional[str] = None, benchmark_term: Optional[str] = None
        ) -> pd.DataFrame:
            body = build_payload(start_date=start_date, end_date=end_date, limit=5000, offset=offset, cusip=cusip, benchmark_term=benchmark_term)
            res = await _post_with_retry(client, json_body=body)
            rows = json.loads(res["returnBody"]["data"]) if res["returnBody"]["data"] else []
            return pd.DataFrame(rows)

        async def _probe_total_for_key(
            client: httpx.AsyncClient, *, start_date, end_date, cusip: Optional[str] = None, benchmark_term: Optional[str] = None
        ) -> tuple[str, int]:
            body = build_payload(start_date=start_date, end_date=end_date, limit=1, offset=1, cusip=cusip, benchmark_term=benchmark_term)
            res = await _post_with_retry(client, json_body=body)
            total = int(res["returnBody"]["headers"]["Record-Total"][0])
            key = cusip if cusip else benchmark_term
            return key, total

        async def _probe_totals(
            client: httpx.AsyncClient,
            *,
            start_date,
            end_date,
            cusips: Optional[List[str]] = None,
            benchmark_terms: Optional[List[str]] = None,
            max_in_flight: int = 4,
        ) -> Dict[str, int]:
            sem = asyncio.Semaphore(max_in_flight)

            async def _task_c(c):
                async with sem:
                    return await _probe_total_for_key(client, start_date=start_date, end_date=end_date, cusip=c)

            async def _task_b(b):
                async with sem:
                    return await _probe_total_for_key(client, start_date=start_date, end_date=end_date, benchmark_term=b)

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

        async def _fetch_all_pages_for_key(
            client: httpx.AsyncClient, *, start_date, end_date, key_is_cusip: bool, key_value: str, total_records: int, max_in_flight: int = 3
        ) -> pd.DataFrame:
            sem = asyncio.Semaphore(max_in_flight)
            pages = math.ceil(total_records / 5000)
            offsets = [i * 5000 for i in range(pages)]  # exact; no +1

            async def _task(off):
                async with sem:
                    return await _fetch_one_page(
                        client,
                        start_date=start_date,
                        end_date=end_date,
                        offset=off,
                        cusip=key_value if key_is_cusip else None,
                        benchmark_term=key_value if not key_is_cusip else None,
                    )

            frames = await asyncio.gather(*[_task(off) for off in offsets], return_exceptions=True)
            dfs = [fr for fr in frames if isinstance(fr, pd.DataFrame) and not fr.empty]
            if not dfs:
                return pd.DataFrame()
            return pd.concat(dfs).sort_values(["tradeDate", "tradeTime"]).reset_index(drop=True)

        async def _run(proxy_url: Optional[str]):
            async with await _build_h2_client(proxy_url) as client:
                warm = await client.get(COOKIE_URL)  # cookie/XSRF warm-up
                warm.raise_for_status()

                t1 = time.time()
                totals = await _probe_totals(
                    client, start_date=start_date, end_date=end_date, cusips=cusips, benchmark_terms=benchmark_terms, max_in_flight=max_in_flight
                )
                self._logger.info(f"TRACE(h2) - FINRA totals probe took: {time.time()-t1:.3f}s")
                self._logger.debug(f"TRACE(h2) - totals: {totals}")

                fetch_tasks = []
                if cusips:
                    for c in cusips:
                        tot = int(totals.get(c, 0))
                        if tot <= 0:
                            self._logger.debug(f"TRACE(h2) - {c} has no records or probe failed.")
                            continue
                        fetch_tasks.append(
                            _fetch_all_pages_for_key(
                                client, start_date=start_date, end_date=end_date, key_is_cusip=True, key_value=c, total_records=tot, max_in_flight=max_in_flight
                            )
                        )
                elif benchmark_terms:
                    for b in benchmark_terms:
                        tot = int(totals.get(b, 0))
                        if tot <= 0:
                            self._logger.debug(f"TRACE(h2) - {b} has no records or probe failed.")
                            continue
                        fetch_tasks.append(
                            _fetch_all_pages_for_key(
                                client, start_date=start_date, end_date=end_date, key_is_cusip=False, key_value=b, total_records=tot, max_in_flight=max_in_flight
                            )
                        )
                else:
                    raise ValueError("Provide 'cusips' or 'benchmark_terms'.")

                fetch_all_t1 = time.time()
                frames = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                self._logger.info(f"TRACE(h2) - Fetch All Took: {time.time()-fetch_all_t1:.3f}s")

                concatenated_dfs: Dict[str, pd.DataFrame] = {}
                idx = 0
                if cusips:
                    for c in cusips:
                        tot = int(totals.get(c, 0))
                        if tot <= 0:
                            continue
                        fr = frames[idx]
                        idx += 1
                        if isinstance(fr, Exception) or fr is None or fr.empty:
                            continue
                        concatenated_dfs[c] = fr
                else:
                    for b in benchmark_terms or []:
                        tot = int(totals.get(b, 0))
                        if tot <= 0:
                            continue
                        fr = frames[idx]
                        idx += 1
                        if isinstance(fr, Exception) or fr is None or fr.empty:
                            continue
                        concatenated_dfs[b] = fr

                if xlsx_path and concatenated_dfs:
                    t1 = time.time()
                    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                        for key, df in concatenated_dfs.items():
                            df.to_excel(writer, sheet_name=str(key)[:31], index=False)
                    self._logger.info(f"TRACE(h2) - XLSX Write Took: {time.time()-t1:.3f}s")

                return concatenated_dfs

        # ---- run with sticky POP; rotate once if the whole batch fails -------
        def _sync_run(proxy_url: Optional[str]):
            return asyncio.run(_run(proxy_url))

        results = None
        try:
            proxy_url = self._get_sticky_proxy_url(ttl=90, rotate=False)
            results = _sync_run(proxy_url)
        except Exception as e1:
            self._logger.warning(f"TRACE(h2) - batch failed on POP. Rotating… ({e1})")
            try:
                proxy_url = self._get_sticky_proxy_url(ttl=90, rotate=True)
                results = _sync_run(proxy_url)
            except Exception as e2:
                self._logger.error(f"TRACE(h2) - batch failed after rotate: {e2}")
                raise

        self._logger.info(f"TRACE(h2) - Total Time Elapsed: {time.time() - total_t1:.3f}s")
        return results or {}

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
