import sqlite3
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "crypto_data.db"

UPBIT_BASE_URL = "https://api.upbit.com/v1"

# 스테이블코인 제외 블랙리스트
STABLECOINS = {
    "KRW-USDT",
    "KRW-USDC",
    "KRW-DAI",
    "KRW-TUSD",
    "KRW-BUSD",
    "KRW-USDE",
    "KRW-FDUSD",
    "KRW-EURC",
}

# DB에 장기 보관할 1년 치 타임프레임 목표치
TIMEFRAME_TARGETS = {
    "1d": 365,
    "240m": 2190,
    "60m": 8760,
}

MINUTE_ENDPOINTS = {
    "1m": "minutes/1",
    "5m": "minutes/5",
    "15m": "minutes/15",
    "60m": "minutes/60",
    "240m": "minutes/240",
}


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
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
            PRIMARY KEY (market, timeframe, candle_time_kst)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_candles_lookup
        ON candles (market, timeframe, candle_time_kst)
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_at TEXT NOT NULL,
            market TEXT NOT NULL,
            current_price REAL NOT NULL,
            rec_buy_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            target_3 REAL NOT NULL,
            target_5 REAL NOT NULL,
            target_10 REAL NOT NULL,
            target_20 REAL NOT NULL,
            prob_3 REAL NOT NULL,
            prob_5 REAL NOT NULL,
            prob_10 REAL NOT NULL,
            prob_20 REAL NOT NULL,
            expected_buy_window TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            verified_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _request_json(url: str, params: dict, max_retries: int = 4):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                time.sleep(0.6 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == max_retries - 1:
                return []
            time.sleep(0.4 * (attempt + 1))
    return []


def get_krw_markets() -> list:
    data = _request_json(f"{UPBIT_BASE_URL}/market/all", {"isDetails": "false"})
    if not data:
        return []
    markets = [
        item["market"]
        for item in data
        if item.get("market", "").startswith("KRW-") and item.get("market") not in STABLECOINS
    ]
    return sorted(markets)


def fetch_candles(
    market: str,
    timeframe: str = "60m",
    count: int = 200,
    to: Optional[str] = None,
) -> list:
    """
    업비트 캔들 조회 (최신순 리스트 반환)
    timeframe: '1d', '240m', '60m', '15m', '5m', '1m'
    """
    count = max(1, min(200, int(count)))
    params = {"market": market, "count": count}
    if to:
        params["to"] = to

    if timeframe == "1d":
        url = f"{UPBIT_BASE_URL}/candles/days"
    elif timeframe in MINUTE_ENDPOINTS:
        url = f"{UPBIT_BASE_URL}/candles/{MINUTE_ENDPOINTS[timeframe]}"
    else:
        raise ValueError(f"지원하지 않는 timeframe: {timeframe}")

    data = _request_json(url, params)
    return data if isinstance(data, list) else []


def fetch_realtime_candles(market: str, timeframe: str = "15m", count: int = 100) -> pd.DataFrame:
    """
    실시간 분석용 캔들 조회 후 시간순(오름차순) DataFrame 반환
    timeframe: '1m', '5m', '15m', '60m', '240m', '1d'
    """
    raw = fetch_candles(market, timeframe=timeframe, count=min(200, count))
    if not raw:
        return pd.DataFrame(columns=["candle_time_kst", "open", "high", "low", "close", "volume"])

    rows = [
        {
            "candle_time_kst": c["candle_date_time_kst"],
            "open": float(c["opening_price"]),
            "high": float(c["high_price"]),
            "low": float(c["low_price"]),
            "close": float(c["trade_price"]),
            "volume": float(c.get("candle_acc_trade_volume", c.get("candle_acc_volume", 0.0))),
        }
        for c in raw
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset=["candle_time_kst"])
    return df.sort_values("candle_time_kst").reset_index(drop=True)


def get_existing_count(market: str, timeframe: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM candles WHERE market = ? AND timeframe = ?",
        (market, timeframe),
    )
    cnt = cur.fetchone()[0]
    conn.close()
    return int(cnt)


def collect_timeframe_for_all(timeframe: str, target_count: int):
    markets = get_krw_markets()
    total = len(markets)
    start_ts = time.time()

    print(f"\n=== [{timeframe}] 수집 시작 (목표: 코인당 {target_count}개) ===")

    for idx, market in enumerate(markets, start=1):
        existing = get_existing_count(market, timeframe)
        if existing >= target_count:
            print(f"[{timeframe} {idx}/{total}] {market} | 이미 {existing}개 보유 (스킵)")
            continue

        to_dt = None
        fetched_total = 0
        seen_times = set()

        while existing + fetched_total < target_count:
            candles = fetch_candles(market, timeframe=timeframe, count=200, to=to_dt)
            if not candles:
                break

            new_rows = []
            oldest_in_batch = None
            for c in candles:
                t_kst = c["candle_date_time_kst"]
                if t_kst in seen_times:
                    continue
                seen_times.add(t_kst)
                oldest_in_batch = t_kst
                new_rows.append((
                    market,
                    timeframe,
                    t_kst,
                    float(c["opening_price"]),
                    float(c["high_price"]),
                    float(c["low_price"]),
                    float(c["trade_price"]),
                    float(c.get("candle_acc_trade_volume", c.get("candle_acc_volume", 0.0))),
                ))

            if not new_rows:
                break

            conn = sqlite3.connect(DB_PATH)
            conn.executemany(
                """
                INSERT OR IGNORE INTO candles (
                    market, timeframe, candle_time_kst, open, high, low, close, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                new_rows,
            )
            conn.commit()
            conn.close()

            fetched_total += len(new_rows)
            if len(candles) < 200 or not oldest_in_batch:
                break
            to_dt = oldest_in_batch
            time.sleep(0.12)

        now_cnt = get_existing_count(market, timeframe)
        elapsed = time.time() - start_ts
        rem_sec = int((elapsed / idx) * (total - idx))
        rem_str = f"{rem_sec // 60}분 {rem_sec % 60}초"
        tag = " (상장 이후 전량)" if now_cnt < target_count else ""
        print(
            f"[{timeframe} {idx}/{total}] {market} | 보유: {now_cnt}개 | "
            f"최과거: {to_dt or '최신'}{tag} | 남은시간 약 {rem_str}"
        )


def run_collection_pipeline():
    init_db()
    collect_timeframe_for_all("1d", TIMEFRAME_TARGETS["1d"])
    collect_timeframe_for_all("240m", TIMEFRAME_TARGETS["240m"])
    collect_timeframe_for_all("60m", TIMEFRAME_TARGETS["60m"])
    print("\n[완료] 1d / 240m / 60m 수집 파이프라인 종료")


if __name__ == "__main__":
    run_collection_pipeline()