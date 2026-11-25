import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "crypto_data.db"
STATE_PATH = BASE_DIR / "collect_state.json"

UPBIT_BASE_URL = "https://api.upbit.com"
UPBIT_MARKETS_URL = f"{UPBIT_BASE_URL}/v1/market/all"
UPBIT_CANDLE_URLS = {
    "60m": f"{UPBIT_BASE_URL}/v1/candles/minutes/60",
    "240m": f"{UPBIT_BASE_URL}/v1/candles/minutes/240",
    "1d": f"{UPBIT_BASE_URL}/v1/candles/days",
}

STABLECOINS: set[str] = {
    "KRW-USDT",
    "KRW-USDC",
    "KRW-DAI",
    "KRW-BUSD",
    "KRW-FDUSD",
    "KRW-PYUSD",
    "KRW-TUSD",
}

TIMEFRAME_TARGETS: dict[str, int] = {
    "1d": 730,
    "240m": 2190,
    "60m": 8760,
}

MAX_CANDLES_PER_REQUEST = 200
MAX_BACKFILL_PAGES_PER_COIN = 5
REQUEST_SLEEP_SECONDS = 0.16
MAX_429_RETRIES = 4


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _ensure_candles_columns(conn: sqlite3.Connection) -> None:
    cols = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(candles)").fetchall()
    }
    if "candle_acc_trade_price" not in cols:
        conn.execute(
            "ALTER TABLE candles ADD COLUMN candle_acc_trade_price REAL DEFAULT 0"
        )


def init_db(conn: sqlite3.Connection | None = None, db_path: Path = DB_PATH) -> None:
    owns_conn = conn is None
    active_conn = conn if conn is not None else get_connection(db_path)
    try:
        active_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                market TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                candle_time_kst TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                candle_acc_trade_price REAL DEFAULT 0,
                PRIMARY KEY (market, timeframe, candle_time_kst)
            )
            """
        )
        _ensure_candles_columns(active_conn)
        active_conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_candles_tf_market_time
            ON candles (timeframe, market, candle_time_kst)
            """
        )
        active_conn.commit()
    finally:
        if owns_conn:
            active_conn.close()


def load_state(state_path: Path = STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state: dict[str, Any], state_path: Path = STATE_PATH) -> None:
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _latest_candle_boundary(timeframe: str, dt: datetime) -> datetime:
    """업비트 KST 기준 해당 시각이 속한 최신 캔들 시작 시각 반환"""
    base = dt.replace(minute=0, second=0, microsecond=0)

    if timeframe == "60m":
        return base

    if timeframe == "240m":
        # 업비트 4시간봉 시작 시각: 01, 05, 09, 13, 17, 21시
        if base.hour == 0:
            return (base - timedelta(days=1)).replace(hour=21)
        shifted = base - timedelta(hours=1)
        slot_start_hour = (shifted.hour // 4) * 4 + 1
        return shifted.replace(hour=slot_start_hour)

    if timeframe == "1d":
        # 업비트 일봉 시작 시각: 매일 09:00
        today_9am = base.replace(hour=9)
        if dt < today_9am:
            return today_9am - timedelta(days=1)
        return today_9am

    return base


def _should_skip_timeframe(
    timeframe: str,
    state: dict[str, Any],
    now_dt: datetime,
) -> tuple[bool, str]:
    """마지막 수집 이후 업비트 기준 새 캔들 경계를 넘지 않았으면 스킵"""
    entry = state.get(timeframe)
    if not isinstance(entry, dict):
        return False, ""

    last_str = entry.get("last_collected_at")
    if not last_str:
        return False, ""

    try:
        last_dt = datetime.fromisoformat(last_str)
    except Exception:
        return False, ""

    cur_boundary = _latest_candle_boundary(timeframe, now_dt)
    last_boundary = _latest_candle_boundary(timeframe, last_dt)

    if cur_boundary <= last_boundary:
        return (
            True,
            f"현재 봉({cur_boundary.strftime('%Y-%m-%d %H:%M')}) 이미 수집됨 "
            f"(마지막 수집: {last_str})",
        )

    return False, ""


def get_krw_markets(
    session: requests.Session | None = None,
    exclude_stablecoins: bool = False,
) -> list[str]:
    client = session or requests.Session()
    resp = client.get(UPBIT_MARKETS_URL, params={"isDetails": "false"}, timeout=10)
    resp.raise_for_status()
    markets = resp.json()
    krw_markets = [
        item["market"]
        for item in markets
        if isinstance(item, dict) and str(item.get("market", "")).startswith("KRW-")
    ]
    if exclude_stablecoins:
        krw_markets = [m for m in krw_markets if m not in STABLECOINS]
    return krw_markets


fetch_krw_markets = get_krw_markets


def _normalize_candle_time(raw_kst: str) -> str:
    # 업비트 응답 예: "2026-11-25T13:00:00"
    return raw_kst[:19]


def fetch_candles(
    market: str,
    timeframe: str,
    to_kst: str | None = None,
    count: int = MAX_CANDLES_PER_REQUEST,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    url = UPBIT_CANDLE_URLS[timeframe]
    params: dict[str, Any] = {
        "market": market,
        "count": min(count, MAX_CANDLES_PER_REQUEST),
    }
    if to_kst:
        params["to"] = to_kst

    client = session or requests.Session()
    for attempt in range(MAX_429_RETRIES + 1):
        resp = client.get(url, params=params, timeout=15)
        if resp.status_code == 429 and attempt < MAX_429_RETRIES:
            retry_after = resp.headers.get("Retry-After")
            try:
                wait_sec = float(retry_after) if retry_after else 0.5 * (attempt + 1)
            except ValueError:
                wait_sec = 0.5 * (attempt + 1)
            time.sleep(max(wait_sec, 0.25))
            continue
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    return []


fetch_candles_page = fetch_candles


def fetch_realtime_candles(
    market: str,
    timeframe: str = "60m",
    count: int = MAX_CANDLES_PER_REQUEST,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """업비트 실시간/최신 캔들을 시간순 DataFrame으로 반환"""
    raw_rows = fetch_candles(
        market=market,
        timeframe=timeframe,
        to_kst=None,
        count=count,
        session=session,
    )
    if not raw_rows:
        return pd.DataFrame(
            columns=[
                "candle_time_kst",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "candle_acc_trade_price",
            ]
        )

    records = [
        {
            "candle_time_kst": _normalize_candle_time(str(item["candle_date_time_kst"])),
            "open": float(item["opening_price"]),
            "high": float(item["high_price"]),
            "low": float(item["low_price"]),
            "close": float(item["trade_price"]),
            "volume": float(item["candle_acc_trade_volume"]),
            "candle_acc_trade_price": float(item.get("candle_acc_trade_price", 0.0)),
        }
        for item in raw_rows
    ]
    df = pd.DataFrame(records)
    df = (
        df.sort_values("candle_time_kst")
        .drop_duplicates("candle_time_kst", keep="last")
        .reset_index(drop=True)
    )
    return df


def upsert_candles(
    conn: sqlite3.Connection,
    market: str,
    timeframe: str,
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0

    _ensure_candles_columns(conn)

    records = []
    for item in rows:
        candle_time = _normalize_candle_time(str(item["candle_date_time_kst"]))
        records.append(
            (
                market,
                timeframe,
                candle_time,
                float(item["opening_price"]),
                float(item["high_price"]),
                float(item["low_price"]),
                float(item["trade_price"]),
                float(item["candle_acc_trade_volume"]),
                float(item.get("candle_acc_trade_price", 0.0)),
            )
        )

    conn.executemany(
        """
        INSERT OR REPLACE INTO candles (
            market, timeframe, candle_time_kst,
            open, high, low, close, volume, candle_acc_trade_price
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()
    return len(records)


def get_market_candle_count(conn: sqlite3.Connection, market: str, timeframe: str) -> int:
    cur = conn.execute(
        "SELECT COUNT(*) FROM candles WHERE market = ? AND timeframe = ?",
        (market, timeframe),
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def get_oldest_candle_time(
    conn: sqlite3.Connection,
    market: str,
    timeframe: str,
) -> str | None:
    cur = conn.execute(
        "SELECT MIN(candle_time_kst) FROM candles WHERE market = ? AND timeframe = ?",
        (market, timeframe),
    )
    row = cur.fetchone()
    return str(row[0]) if row and row[0] else None


def collect_market_timeframe(
    conn: sqlite3.Connection,
    market: str,
    timeframe: str,
    target_count: int,
    session: requests.Session,
) -> int:
    total_saved = 0

    # 1. 최신 캔들 수집
    latest_rows = fetch_candles(
        market=market,
        timeframe=timeframe,
        to_kst=None,
        count=MAX_CANDLES_PER_REQUEST,
        session=session,
    )
    total_saved += upsert_candles(conn, market, timeframe, latest_rows)
    time.sleep(REQUEST_SLEEP_SECONDS)

    # 2. 목표 개수 미만이면 과거 구간 백필(최대 MAX_BACKFILL_PAGES_PER_COIN 페이지)
    pages_fetched = 0
    while (
        get_market_candle_count(conn, market, timeframe) < target_count
        and pages_fetched < MAX_BACKFILL_PAGES_PER_COIN
    ):
        oldest_time = get_oldest_candle_time(conn, market, timeframe)
        if not oldest_time:
            break

        older_rows = fetch_candles(
            market=market,
            timeframe=timeframe,
            to_kst=oldest_time,
            count=MAX_CANDLES_PER_REQUEST,
            session=session,
        )
        if not older_rows:
            break

        total_saved += upsert_candles(conn, market, timeframe, older_rows)
        pages_fetched += 1
        time.sleep(REQUEST_SLEEP_SECONDS)

    return total_saved


def collect_timeframe_for_all(
    timeframe: str,
    target_count: int | None = None,
    force: bool = False,
    db_path: Path = DB_PATH,
    state_path: Path = STATE_PATH,
) -> dict[str, Any]:
    if timeframe not in UPBIT_CANDLE_URLS:
        raise ValueError(f"지원하지 않는 타임프레임입니다: {timeframe}")

    resolved_target = target_count or TIMEFRAME_TARGETS.get(timeframe, 1000)
    now_dt = datetime.now()
    state = load_state(state_path)

    if not force:
        should_skip, reason = _should_skip_timeframe(timeframe, state, now_dt)
        if should_skip:
            print(f"[{timeframe}] 스킵: {reason}")
            return {"timeframe": timeframe, "skipped": True, "reason": reason}

    conn = get_connection(db_path)
    init_db(conn)

    session = requests.Session()
    try:
        markets = get_krw_markets(session, exclude_stablecoins=True)
        print(f"[{timeframe}] 대상 코인 {len(markets)}개 수집 시작 (목표 {resolved_target}개)")

        total_rows = 0
        for idx, market in enumerate(markets, start=1):
            before_cnt = get_market_candle_count(conn, market, timeframe)
            try:
                collect_market_timeframe(
                    conn=conn,
                    market=market,
                    timeframe=timeframe,
                    target_count=resolved_target,
                    session=session,
                )
            except Exception as exc:
                print(f"[{timeframe}] {market} 수집 오류: {exc}")
                continue

            after_cnt = get_market_candle_count(conn, market, timeframe)
            diff = after_cnt - before_cnt
            total_rows += max(diff, 0)

            if idx % 50 == 0 or idx == len(markets):
                print(f"[{timeframe}] 진행 {idx}/{len(markets)} ({market}: +{diff}개, 누적 {after_cnt}개)")

        finished_at = datetime.now().isoformat(timespec="seconds")
        state[timeframe] = {
            "last_collected_at": finished_at,
            "market_count": len(markets),
            "target_count": resolved_target,
        }
        save_state(state, state_path)

        print(f"[{timeframe}] 완료: 총 +{total_rows}개 반영 ({finished_at})")
        return {
            "timeframe": timeframe,
            "skipped": False,
            "market_count": len(markets),
            "added_rows": total_rows,
            "finished_at": finished_at,
        }
    finally:
        session.close()
        conn.close()


def run_collection_pipeline(
    timeframes: tuple[str, ...] = ("1d", "240m", "60m"),
    force: bool = False,
    db_path: Path = DB_PATH,
    state_path: Path = STATE_PATH,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for tf in timeframes:
        results[tf] = collect_timeframe_for_all(
            timeframe=tf,
            target_count=TIMEFRAME_TARGETS.get(tf),
            force=force,
            db_path=db_path,
            state_path=state_path,
        )
    return results


if __name__ == "__main__":
    run_collection_pipeline()