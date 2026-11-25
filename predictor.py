import sqlite3
import time
import warnings
from bisect import bisect_right
from datetime import datetime, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from collector import (
    BASE_DIR,
    DB_PATH,
    STABLECOINS,
    fetch_candles,
    fetch_realtime_candles,
    get_krw_markets,
    init_db,
)

warnings.filterwarnings("ignore")

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGETS = [3, 5, 10, 20]      # +3%, +5%, +10%, +20%
HORIZON_CANDLES = 24          # 학습 라벨 기준(60분봉 24개)
HOLD_HOURS_BY_STYLE = {
    "단타": 6.0,              # 단타 검증 기한(6시간)
    "스윙": 48.0,             # 스윙 검증 기한(48시간)
    "안정스윙": 48.0,         # 5~10% 안정스윙 검증 기한(48시간)
    "중기추세": 72.0,         # 10~20% 중기추세 검증 기한(72시간)
}
STOP_LOSS_BY_TARGET = {
    3: 0.025,
    5: 0.030,
    10: 0.040,
    20: 0.050,
}


def snap_to_upbit_tick(price: float) -> float:
    """업비트 KRW 마켓 가격대별 호가 단위에 맞춰 스냅"""
    if price is None or pd.isna(price) or float(price) <= 0:
        return 0.0
    p = float(price)

    if p >= 2_000_000:
        tick = 1000.0
    elif p >= 1_000_000:
        tick = 500.0
    elif p >= 500_000:
        tick = 500.0
    elif p >= 100_000:
        tick = 100.0
    elif p >= 50_000:
        tick = 50.0
    elif p >= 10_000:
        tick = 10.0
    elif p >= 1_000:
        tick = 5.0
    elif p >= 100:
        tick = 1.0
    elif p >= 10:
        tick = 0.1
    elif p >= 1:
        tick = 0.01
    else:
        tick = 0.0001

    snapped = round(p / tick) * tick
    if tick >= 1:
        return float(int(round(snapped)))
    decimals = len(str(tick).split(".")[1]) if "." in str(tick) else 4
    return float(round(snapped, decimals))


def _connect_db(timeout: float = 30.0) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=timeout, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except sqlite3.OperationalError:
        pass
    return conn


def _exec_write(sql_or_fn, params=(), max_retries: int = 8):
    last_exc = None
    for attempt in range(max_retries):
        try:
            conn = _connect_db(timeout=30.0)
            try:
                if callable(sql_or_fn):
                    res = sql_or_fn(conn)
                else:
                    res = conn.execute(sql_or_fn, params)
                conn.commit()
                return res
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "locked" in msg or "busy" in msg:
                time.sleep(0.15 * (attempt + 1))
                continue
            raise
    if last_exc:
        raise last_exc


def ensure_log_columns():
    """기존 prediction_logs 테이블에 통계/대응유형 컬럼이 없으면 안전하게 추가"""
    try:
        init_db()
    except Exception as exc:
        print(f"[WARN] init_db skipped in ensure_log_columns: {exc}")

    def _alter(conn):
        cur = conn.cursor()
        tbl = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_logs'"
        ).fetchone()
        if not tbl:
            return

        cur.execute("PRAGMA table_info(prediction_logs)")
        existing_cols = {row[1] for row in cur.fetchall()}

        alter_stmts = [
            ("max_gain_pct", "REAL DEFAULT 0.0"),
            ("hit_3", "INTEGER DEFAULT 0"),
            ("hit_5", "INTEGER DEFAULT 0"),
            ("hit_10", "INTEGER DEFAULT 0"),
            ("hit_20", "INTEGER DEFAULT 0"),
            ("outcome", "TEXT DEFAULT 'PENDING'"),   # PENDING / SUCCESS / FAIL
            ("trade_style", "TEXT DEFAULT '단타'"),  # 단타 / 스윙 / 안정스윙 / 중기추세
        ]
        for col_name, col_def in alter_stmts:
            if col_name not in existing_cols:
                try:
                    cur.execute(f"ALTER TABLE prediction_logs ADD COLUMN {col_name} {col_def}")
                except sqlite3.OperationalError as col_exc:
                    print(f"[WARN] skip ALTER COLUMN {col_name}: {col_exc}")

    try:
        _exec_write(_alter)
    except Exception as exc:
        print(f"[WARN] ensure_log_columns skipped: {exc}")


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_60m_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]

    d["ret_1h"] = close.pct_change(1)
    d["ret_3h"] = close.pct_change(3)
    d["ret_6h"] = close.pct_change(6)
    d["ret_12h"] = close.pct_change(12)
    d["ret_24h"] = close.pct_change(24)

    candle_range = (d["high"] - d["low"]).replace(0, np.nan)
    d["body_ratio"] = ((close - d["open"]) / candle_range).fillna(0.0)
    d["upper_wick_ratio"] = ((d["high"] - np.maximum(close, d["open"])) / candle_range).fillna(0.0)
    d["range_pct"] = (d["high"] - d["low"]) / close.replace(0, np.nan)

    for w in [20, 60, 120]:
        ema = close.ewm(span=w, adjust=False).mean()
        d[f"ema_{w}_dist"] = (close - ema) / ema.replace(0, np.nan)

    d["rsi_14"] = compute_rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = (macd - signal) / close.replace(0, np.nan)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_span = (bb_upper - bb_lower).replace(0, np.nan)
    d["bb_pos"] = ((close - bb_lower) / bb_span).clip(-0.5, 1.5).fillna(0.5)
    d["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    atr = compute_atr(d, 14)
    d["atr_pct"] = atr / close.replace(0, np.nan)
    vol_ma20 = d["volume"].rolling(20).mean().replace(0, np.nan)
    d["vol_ratio"] = (d["volume"] / vol_ma20).clip(0, 20).fillna(1.0)
    d["vol_trend_3h"] = (d["volume"].rolling(3).mean() / vol_ma20).clip(0, 20).fillna(1.0)

    safe_atr = d["atr_pct"].replace(0, np.nan).fillna(0.01)
    d["ret_1h_atr"] = (d["ret_1h"] / safe_atr).clip(-10, 10).fillna(0.0)
    d["ret_6h_atr"] = (d["ret_6h"] / safe_atr).clip(-15, 15).fillna(0.0)
    high_24 = d["high"].rolling(24).max().replace(0, np.nan)
    d["dist_high_24h"] = ((close - high_24) / high_24).clip(-0.5, 0.2).fillna(0.0)

    hour = pd.to_datetime(d["candle_time_kst"]).dt.hour
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    return d


def load_daily_context_map() -> dict:
    conn = _connect_db(timeout=30.0)
    try:
        df_d = pd.read_sql_query(
            """
            SELECT market, candle_time_kst, close
            FROM candles
            WHERE timeframe = '1d'
            ORDER BY market, candle_time_kst
            """,
            conn,
        )
    finally:
        conn.close()

    context_map = {}
    if df_d.empty:
        return context_map

    for market, grp in df_d.groupby("market"):
        g = grp.sort_values("candle_time_kst").copy()
        ema20 = g["close"].ewm(span=20, adjust=False).mean()
        ema60 = g["close"].ewm(span=60, adjust=False).mean()
        g["d_ema20_dist"] = ((g["close"] - ema20) / ema20.replace(0, np.nan)).shift(1)
        g["d_ema60_dist"] = ((g["close"] - ema60) / ema60.replace(0, np.nan)).shift(1)
        g["d_rsi_14"] = compute_rsi(g["close"], 14).shift(1)
        g["date_str"] = g["candle_time_kst"].str.slice(0, 10)

        for _, row in g.dropna(subset=["d_ema20_dist"]).iterrows():
            context_map[(market, row["date_str"])] = {
                "d_ema20_dist": float(row["d_ema20_dist"]),
                "d_ema60_dist": float(row["d_ema60_dist"]),
                "d_rsi_14": float(row["d_rsi_14"]),
            }
    return context_map


def load_h4_context_map() -> dict:
    conn = _connect_db(timeout=30.0)
    try:
        df_h4 = pd.read_sql_query(
            """
            SELECT market, candle_time_kst, close
            FROM candles
            WHERE timeframe = '240m'
            ORDER BY market, candle_time_kst
            """,
            conn,
        )
    finally:
        conn.close()

    h4_map = {}
    if df_h4.empty:
        return h4_map

    for market, grp in df_h4.groupby("market"):
        g = grp.sort_values("candle_time_kst").copy()
        ema20 = g["close"].ewm(span=20, adjust=False).mean()
        ema60 = g["close"].ewm(span=60, adjust=False).mean()

        g["h4_ema20_dist"] = ((g["close"] - ema20) / ema20.replace(0, np.nan)).shift(1)
        g["h4_ema60_dist"] = ((g["close"] - ema60) / ema60.replace(0, np.nan)).shift(1)
        g["h4_rsi_14"] = compute_rsi(g["close"], 14).shift(1)
        g["h4_ret_24h"] = g["close"].pct_change(6).shift(1)

        g = g.dropna(subset=["h4_ema20_dist"])
        times = g["candle_time_kst"].tolist()
        records = [
            {
                "h4_ema20_dist": float(r["h4_ema20_dist"]),
                "h4_ema60_dist": float(r["h4_ema60_dist"]),
                "h4_rsi_14": float(r["h4_rsi_14"]) if pd.notna(r["h4_rsi_14"]) else 50.0,
                "h4_ret_24h": float(r["h4_ret_24h"]) if pd.notna(r["h4_ret_24h"]) else 0.0,
            }
            for _, r in g.iterrows()
        ]
        h4_map[market] = (times, records)

    return h4_map


def _lookup_h4(h4_entry, ts_str: str) -> dict:
    if not h4_entry:
        return {}
    times, records = h4_entry
    idx = bisect_right(times, ts_str) - 1
    if idx < 0:
        return {}
    return records[idx]


def load_btc_60m_map() -> dict:
    conn = _connect_db(timeout=30.0)
    try:
        btc = pd.read_sql_query(
            """
            SELECT candle_time_kst, close
            FROM candles
            WHERE market = 'KRW-BTC' AND timeframe = '60m'
            ORDER BY candle_time_kst
            """,
            conn,
        )
    finally:
        conn.close()

    btc_map = {}
    if btc.empty:
        return btc_map

    btc = btc.sort_values("candle_time_kst").copy()
    btc["btc_ret_1h"] = btc["close"].pct_change(1)
    btc["btc_ret_6h"] = btc["close"].pct_change(6)
    btc["btc_rsi_14"] = compute_rsi(btc["close"], 14)

    for _, row in btc.iterrows():
        btc_map[row["candle_time_kst"]] = {
            "btc_ret_1h": float(row["btc_ret_1h"]) if pd.notna(row["btc_ret_1h"]) else 0.0,
            "btc_ret_6h": float(row["btc_ret_6h"]) if pd.notna(row["btc_ret_6h"]) else 0.0,
            "btc_rsi_14": float(row["btc_rsi_14"]) if pd.notna(row["btc_rsi_14"]) else 50.0,
        }
    return btc_map


FEATURE_COLS = [
    "ret_1h", "ret_3h", "ret_6h", "ret_12h", "ret_24h",
    "body_ratio", "upper_wick_ratio", "range_pct",
    "ema_20_dist", "ema_60_dist", "ema_120_dist",
    "rsi_14", "macd_hist", "bb_pos", "bb_width", "atr_pct", "vol_ratio",
    "vol_trend_3h", "ret_1h_atr", "ret_6h_atr", "dist_high_24h",
    "hour_sin", "hour_cos",
    "d_ema20_dist", "d_ema60_dist", "d_rsi_14",
    "h4_ema20_dist", "h4_ema60_dist", "h4_rsi_14", "h4_ret_24h",
    "btc_ret_1h", "btc_ret_6h", "btc_rsi_14",
]


def enrich_dataframe(
    df_60m: pd.DataFrame,
    market: str,
    daily_map: dict,
    h4_map: dict,
    btc_map: dict,
) -> pd.DataFrame:
    d = add_60m_indicators(df_60m)
    dates = d["candle_time_kst"].str.slice(0, 10)
    h4_entry = h4_map.get(market)

    d_ema20, d_ema60, d_rsi = [], [], []
    h4_ema20, h4_ema60, h4_rsi, h4_ret24 = [], [], [], []
    b_ret1, b_ret6, b_rsi = [], [], []

    for dt_str, ts_str in zip(dates, d["candle_time_kst"]):
        dc = daily_map.get((market, dt_str), {})
        d_ema20.append(dc.get("d_ema20_dist", 0.0))
        d_ema60.append(dc.get("d_ema60_dist", 0.0))
        d_rsi.append(dc.get("d_rsi_14", 50.0))

        hc = _lookup_h4(h4_entry, ts_str)
        h4_ema20.append(hc.get("h4_ema20_dist", 0.0))
        h4_ema60.append(hc.get("h4_ema60_dist", 0.0))
        h4_rsi.append(hc.get("h4_rsi_14", 50.0))
        h4_ret24.append(hc.get("h4_ret_24h", 0.0))

        bc = btc_map.get(ts_str, {})
        b_ret1.append(bc.get("btc_ret_1h", 0.0))
        b_ret6.append(bc.get("btc_ret_6h", 0.0))
        b_rsi.append(bc.get("btc_rsi_14", 50.0))

    d["d_ema20_dist"] = d_ema20
    d["d_ema60_dist"] = d_ema60
    d["d_rsi_14"] = d_rsi
    d["h4_ema20_dist"] = h4_ema20
    d["h4_ema60_dist"] = h4_ema60
    d["h4_rsi_14"] = h4_rsi
    d["h4_ret_24h"] = h4_ret24
    d["btc_ret_1h"] = b_ret1
    d["btc_ret_6h"] = b_ret6
    d["btc_rsi_14"] = b_rsi

    return d


def classify_trade_style(row: pd.Series, p3: float, p5: float, p10: float, p20: float) -> tuple[str, str]:
    """코인 지표와 목표 확률을 기반으로 단타 vs 스윙 대응 적합도를 판별"""
    atr_pct = float(row.get("atr_pct", 0.015))
    bb_width = float(row.get("bb_width", 0.04))
    ret_1h = float(row.get("ret_1h", 0.0))
    ret_3h = float(row.get("ret_3h", 0.0))
    vol_ratio = float(row.get("vol_ratio", 1.0))
    vol_trend_3h = float(row.get("vol_trend_3h", 1.0))
    rsi_14 = float(row.get("rsi_14", 50.0))

    d_ema20 = float(row.get("d_ema20_dist", 0.0))
    h4_ema20 = float(row.get("h4_ema20_dist", 0.0))
    h4_ema60 = float(row.get("h4_ema60_dist", 0.0))

    short_score = 0.0
    swing_score = 0.0
    reasons = []

    if atr_pct >= 0.020 or bb_width >= 0.065:
        short_score += 1.5
        reasons.append("변동성 큼")
    if vol_ratio >= 1.7 or vol_trend_3h >= 1.5:
        short_score += 1.5
        reasons.append("단기 수급 유입")
    if abs(ret_1h) >= 0.012 or ret_3h >= 0.020:
        short_score += 1.0
        reasons.append("단기 탄력 활성")
    if rsi_14 >= 68.0:
        short_score += 0.5

    if h4_ema20 >= -0.015 and h4_ema60 >= -0.025:
        swing_score += 1.5
        reasons.append("4H 추세 지지")
    if d_ema20 >= -0.025 and 40.0 <= rsi_14 <= 67.0:
        swing_score += 1.5
        reasons.append("일봉 흐름·과열여유")
    if (p10 + p20) >= max(p3 * 0.85, 0.30) and atr_pct < 0.032:
        swing_score += 1.0
        reasons.append("중기 목표 여력")

    style = "스윙" if (swing_score > short_score and (p5 >= 0.40 or p10 >= 0.22)) else "단타"
    reason_text = ", ".join(reasons[:2]) if reasons else ("빠른 회전 유리" if style == "단타" else "추세 호흡 유리")
    return style, reason_text


def classify_swing_track(row: pd.Series, p5: float, p10: float, p20: float) -> tuple[str, str]:
    """5~10% 안정스윙용 vs 10~20% 중기추세용 판별"""
    d_ema20 = float(row.get("d_ema20_dist", 0.0))
    d_ema60 = float(row.get("d_ema60_dist", 0.0))
    h4_ema20 = float(row.get("h4_ema20_dist", 0.0))
    h4_ema60 = float(row.get("h4_ema60_dist", 0.0))
    h4_ret24 = float(row.get("h4_ret_24h", 0.0))
    rsi_14 = float(row.get("rsi_14", 50.0))
    atr_pct = float(row.get("atr_pct", 0.015))

    trend_score = 0.0
    reasons = []

    if h4_ema20 >= -0.01 and h4_ema60 >= -0.02:
        trend_score += 1.5
        reasons.append("4H 추세 지지")
    if d_ema20 >= -0.02 and d_ema60 >= -0.03:
        trend_score += 1.5
        reasons.append("일봉 추세 양호")
    if h4_ret24 >= 0.02 or p20 >= 0.25:
        trend_score += 1.0
        reasons.append("중기 상승 여력")
    if 40.0 <= rsi_14 <= 68.0:
        reasons.append("과열 부담 적음")

    if (p20 >= 0.25 or p10 >= 0.45) and trend_score >= 2.5 and atr_pct <= 0.045:
        reason_text = ", ".join(reasons[:2]) if reasons else "10~20% 중기 추세 우위"
        return "중기추세", reason_text

    reason_text = ", ".join(reasons[:2]) if reasons else "5~10% 안정 스윙 적합"
    return "안정스윙", reason_text


def _compute_clean_target_series(d: pd.DataFrame, target_pct: float, stop_pct: float) -> pd.Series:
    highs = d["high"].to_numpy(dtype=float)
    lows = d["low"].to_numpy(dtype=float)
    closes = d["close"].to_numpy(dtype=float)
    n = len(d)
    labels = np.full(n, np.nan, dtype=float)

    tgt_mult = 1.0 + (target_pct / 100.0)
    stp_mult = 1.0 - max(0.0, float(stop_pct))

    for i in range(n - HORIZON_CANDLES):
        entry = closes[i]
        if entry <= 0:
            continue
        target_price = entry * tgt_mult
        stop_price = entry * stp_mult

        hit_target = False
        for j in range(i + 1, i + 1 + HORIZON_CANDLES):
            if lows[j] <= stop_price:
                break
            if highs[j] >= target_price:
                hit_target = True
                break

        labels[i] = 1.0 if hit_target else 0.0

    return pd.Series(labels, index=d.index)


def build_training_dataset() -> pd.DataFrame:
    conn = _connect_db(timeout=30.0)
    try:
        df_all = pd.read_sql_query(
            """
            SELECT market, candle_time_kst, open, high, low, close, volume
            FROM candles
            WHERE timeframe = '60m'
            ORDER BY market, candle_time_kst
            """,
            conn,
        )
    finally:
        conn.close()

    if df_all.empty:
        raise RuntimeError("DB에 60분봉('60m') 데이터가 없습니다. collector.py를 먼저 실행하세요.")

    daily_map = load_daily_context_map()
    h4_map = load_h4_context_map()
    btc_map = load_btc_60m_map()
    samples = []

    for market, grp in df_all.groupby("market"):
        if market in STABLECOINS or len(grp) < 150:
            continue

        d = enrich_dataframe(grp.sort_values("candle_time_kst"), market, daily_map, h4_map, btc_map)

        future_max_high = (
            d["high"]
            .shift(-1)
            .rolling(window=HORIZON_CANDLES, min_periods=HORIZON_CANDLES)
            .max()
            .shift(-(HORIZON_CANDLES - 1))
        )
        max_future_return_pct = ((future_max_high - d["close"]) / d["close"]) * 100.0

        target_cols = []
        for pct in TARGETS:
            t_col = f"target_{pct}"
            stop_ratio = STOP_LOSS_BY_TARGET.get(pct, 0.030)
            d[t_col] = _compute_clean_target_series(d, target_pct=float(pct), stop_pct=stop_ratio)
            target_cols.append(t_col)

        d["market"] = market
        d["max_future_return_pct"] = max_future_return_pct
        valid = d.dropna(subset=FEATURE_COLS + ["max_future_return_pct"] + target_cols)
        if not valid.empty:
            samples.append(valid)

    if not samples:
        raise RuntimeError("학습 가능한 유효 캔들 데이터가 부족합니다. 수집이 조금 더 진행된 뒤 다시 시도하세요.")

    dataset = pd.concat(samples, ignore_index=True)
    dataset = dataset.sort_values("candle_time_kst").reset_index(drop=True)
    return dataset


def train_models():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 실전형 학습 데이터셋 생성 중...")
    dataset = build_training_dataset()
    print(
        f"총 학습/검증 샘플 수: {len(dataset):,}개 "
        f"(기간: {dataset['candle_time_kst'].min()} ~ {dataset['candle_time_kst'].max()})"
    )

    split_idx = int(len(dataset) * 0.8)
    gap = min(HORIZON_CANDLES, max(0, len(dataset) - split_idx - 1))
    train_df = dataset.iloc[: max(1, split_idx - gap)]
    val_df = dataset.iloc[split_idx:]

    X_train, X_val = train_df[FEATURE_COLS], val_df[FEATURE_COLS]

    for pct in TARGETS:
        col = f"target_{pct}"
        y_train = train_df[col].astype(int)
        y_val = val_df[col].astype(int)

        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=40,
            subsample_for_bin=50000,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.2,
            reg_lambda=1.0,
            scale_pos_weight=min(scale_pos_weight, 8.0),
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)],
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs) if y_val.nunique() > 1 else 0.5
        base_rate = y_val.mean() * 100.0
        high_conf_rate = (
            y_val[val_probs >= 0.6].mean() * 100.0 if (val_probs >= 0.6).sum() > 0 else 0.0
        )

        model_path = MODEL_DIR / f"lgb_target_{pct}.txt"
        model.booster_.save_model(str(model_path))

        print(
            f"  [+{pct}% 모델] 저장완료 | 검증 AUC: {auc:.3f} | "
            f"전체 돌파율: {base_rate:.1f}% | 고확신(>=60%) 실제 돌파율: {high_conf_rate:.1f}%"
        )


def load_trained_models() -> dict:
    models = {}
    for pct in TARGETS:
        model_path = MODEL_DIR / f"lgb_target_{pct}.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}. 먼저 train_models()를 실행하세요.")
        models[pct] = lgb.Booster(model_file=str(model_path))
    return models


def _fetch_live_1m_price(market: str, fallback_price: float) -> float:
    """스캔 시점의 최신 1분봉 종가를 업비트 호가 단위로 반환하고, 실패 시 fallback_price를 스냅해 반환"""
    try:
        df_1m = fetch_realtime_candles(market, timeframe="1m", count=1)
        if df_1m is not None and not df_1m.empty and "close" in df_1m.columns:
            live_p = float(df_1m["close"].iloc[-1])
            if live_p > 0:
                return snap_to_upbit_tick(live_p)
    except Exception:
        pass
    return snap_to_upbit_tick(fallback_price)


def analyze_micro_timing(
    market: str,
    current_price: float,
    atr_pct: float,
    rsi_14: float,
    trade_style: str = "단타",
):
    """단타/스윙 대응유형별로 업비트 호가 스냅이 적용된 추천매수가, 손절가, 예상매수시점을 계산"""
    df_15m = fetch_realtime_candles(market, timeframe="15m", count=96)
    time.sleep(0.08)
    df_5m = fetch_realtime_candles(market, timeframe="5m", count=60)
    time.sleep(0.08)
    df_1m = fetch_realtime_candles(market, timeframe="1m", count=30)

    cur_p = snap_to_upbit_tick(current_price)
    safe_atr_pct = float(np.clip(atr_pct, 0.008, 0.05))
    notes = []

    if trade_style in ("스윙", "안정스윙", "중기추세"):
        if df_15m.empty or len(df_15m) < 20:
            raw_buy = cur_p * (1 - 0.65 * safe_atr_pct)
            rec_buy = snap_to_upbit_tick(raw_buy)
            stop_ratio = float(np.clip(1.85 * safe_atr_pct, 0.028, 0.065))
            stop_loss = snap_to_upbit_tick(rec_buy * (1 - stop_ratio))
            return rec_buy, stop_loss, "약 1 ~ 4시간 내 (15m 눌림 대기)", "스윙 기본 눌림 기준"

        ema20_15m = float(df_15m["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        low_15m = float(df_15m["low"].tail(12).min())
        candidate_buy = min(cur_p * (1 - 0.55 * safe_atr_pct), max(ema20_15m, low_15m))
        raw_buy = float(np.clip(candidate_buy, cur_p * 0.955, cur_p))

        if not df_5m.empty and len(df_5m) >= 10:
            low_5m = float(df_5m["low"].tail(8).min())
            if low_5m < raw_buy:
                raw_buy = max(low_5m, cur_p * 0.955)
                notes.append("5m 지지선 반영")

        rec_buy = snap_to_upbit_tick(raw_buy)
        if rec_buy > cur_p:
            rec_buy = cur_p

        gap_pct = ((cur_p - rec_buy) / cur_p) * 100.0 if cur_p > 0 else 0.0
        if rsi_14 >= 72 or gap_pct >= 1.5:
            expected_window = "약 2 ~ 6시간 내 (스윙 눌림 자리 대기)"
        elif gap_pct >= 0.5:
            expected_window = "약 1 ~ 3시간 내 (눌림목 접근 대기)"
        else:
            expected_window = "당일 내 눌림 구간 분할 진입 유효"

        stop_ratio = float(np.clip(1.85 * safe_atr_pct, 0.028, 0.065))
        stop_loss = snap_to_upbit_tick(rec_buy * (1 - stop_ratio))
        if stop_loss >= rec_buy:
            stop_loss = snap_to_upbit_tick(rec_buy * 0.97)
        detail_memo = " | ".join(notes) if notes else "스윙 눌림·추세 기준"
        return rec_buy, stop_loss, expected_window, detail_memo

    if df_15m.empty or len(df_15m) < 20:
        raw_buy = cur_p * (1 - float(np.clip(0.35 * safe_atr_pct, 0.002, 0.015)))
        rec_buy = snap_to_upbit_tick(raw_buy)
        stop_ratio = float(np.clip(1.15 * safe_atr_pct, 0.015, 0.025))
        stop_loss = snap_to_upbit_tick(rec_buy * (1 - stop_ratio))
        return rec_buy, stop_loss, "즉시 ~ 30분 이내", "단타 기본 기준"

    ema20_15m = float(df_15m["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    recent_low_15m = float(df_15m["low"].tail(6).min())
    candidate_buy = min(cur_p * (1 - 0.25 * safe_atr_pct), max(ema20_15m, recent_low_15m))
    raw_buy = float(np.clip(candidate_buy, cur_p * 0.985, cur_p))

    vol_5m_spike = False
    if not df_5m.empty and len(df_5m) >= 15:
        vol_ma_5m = df_5m["volume"].tail(15).mean()
        last_vol_5m = float(df_5m["volume"].iloc[-1])
        low_5m = float(df_5m["low"].tail(5).min())
        if low_5m < raw_buy:
            raw_buy = max(low_5m, cur_p * 0.985)
        if vol_ma_5m > 0 and last_vol_5m >= vol_ma_5m * 1.8:
            vol_5m_spike = True
            notes.append("5m 거래량 급증")
            if rsi_14 < 70:
                raw_buy = max(raw_buy, cur_p * 0.996)

    spike_1m = False
    if not df_1m.empty and len(df_1m) >= 6:
        prev_p = float(df_1m["close"].iloc[-6])
        if prev_p > 0:
            ret_5m_from_1m = (float(df_1m["close"].iloc[-1]) - prev_p) / prev_p
            if abs(ret_5m_from_1m) >= 0.015:
                spike_1m = True
                notes.append(f"1m 직전 5분 변동 {ret_5m_from_1m * 100:+.1f}%")

    rec_buy = snap_to_upbit_tick(raw_buy)
    if rec_buy > cur_p:
        rec_buy = cur_p

    gap_pct = ((cur_p - rec_buy) / cur_p) * 100.0 if cur_p > 0 else 0.0
    if rsi_14 >= 72 or gap_pct >= 1.0:
        expected_window = "약 30분 ~ 1시간 내 (단기 과열/눌림 대기)"
    elif spike_1m and gap_pct >= 0.25:
        expected_window = "약 10 ~ 30분 내 (1m 급등락 진정 후 진입)"
    elif gap_pct >= 0.35:
        expected_window = "약 15 ~ 45분 내 (얕은 눌림 접근 중)"
    else:
        expected_window = "즉시 ~ 20분 이내 빠른 대응 유효" + (" (5m 수급 강함)" if vol_5m_spike else "")

    stop_ratio = float(np.clip(1.15 * safe_atr_pct, 0.015, 0.025))
    stop_loss = snap_to_upbit_tick(rec_buy * (1 - stop_ratio))
    if stop_loss >= rec_buy:
        stop_loss = snap_to_upbit_tick(rec_buy * 0.985)
    detail_memo = " | ".join(notes) if notes else "단타 기민 대응"
    return rec_buy, stop_loss, expected_window, detail_memo


def _style_targets(rec_buy: float, trade_style: str) -> tuple[float, float, float, float]:
    """대응유형과 업비트 호가 단위에 맞춘 목표가(+3%, +5%, +10%, +20%) 산출"""
    base = snap_to_upbit_tick(rec_buy)
    if trade_style == "단타":
        pcts = (2.2, 4.0, 7.5, 13.0)
    elif trade_style == "중기추세":
        pcts = (5.0, 10.0, 15.0, 20.0)
    elif trade_style == "안정스윙":
        pcts = (3.5, 5.5, 8.0, 10.0)
    else:
        pcts = (3.0, 5.5, 10.5, 20.0)

    targets = []
    prev_t = base
    for pct in pcts:
        t = snap_to_upbit_tick(base * (1.0 + pct / 100.0))
        if t <= prev_t:
            t = snap_to_upbit_tick(prev_t * 1.005)
            if t <= prev_t:
                t = prev_t + (1.0 if base >= 100 else 0.01)
        targets.append(t)
        prev_t = t

    return targets[0], targets[1], targets[2], targets[3]


def scan_market(top_n: int = 10, min_prob_3: float = 0.55, save_to_db: bool = True) -> pd.DataFrame:
    ensure_log_columns()
    models = load_trained_models()
    daily_map = load_daily_context_map()
    h4_map = load_h4_context_map()
    btc_map = load_btc_60m_map()
    markets = [m for m in get_krw_markets() if m not in STABLECOINS]

    conn = _connect_db(timeout=30.0)
    try:
        df_all = pd.read_sql_query(
            """
            SELECT market, candle_time_kst, open, high, low, close, volume
            FROM candles
            WHERE timeframe = '60m'
            ORDER BY market, candle_time_kst
            """,
            conn,
        )
    finally:
        conn.close()

    if df_all.empty:
        return pd.DataFrame()

    grouped_60m = {str(m): grp for m, grp in df_all.groupby("market")}
    feature_batches = []
    meta_batches = []

    for market in markets:
        grp = grouped_60m.get(market)
        if grp is None or len(grp) < 130:
            continue
        d = enrich_dataframe(grp.sort_values("candle_time_kst"), market, daily_map, h4_map, btc_map)
        last_row = d.iloc[-1]
        if last_row[FEATURE_COLS].isna().any():
            continue
        if float(last_row["rsi_14"]) >= 82.0 or float(last_row["ret_3h"]) <= -0.08:
            continue

        feature_batches.append(last_row[FEATURE_COLS].astype(float).values)
        meta_batches.append({
            "market": market,
            "last_row": last_row,
            "candle_time_kst": last_row["candle_time_kst"],
            "current_price": float(last_row["close"]),
            "atr_pct": float(last_row["atr_pct"]),
            "rsi_14": float(last_row["rsi_14"]),
            "vol_ratio": float(last_row["vol_ratio"]),
        })

    if not feature_batches:
        return pd.DataFrame()

    X_latest = np.vstack(feature_batches)
    probs_by_target = {pct: models[pct].predict(X_latest) for pct in TARGETS}

    candidates = []
    for i, meta in enumerate(meta_batches):
        p3 = float(np.clip(probs_by_target[3][i], 0.0, 1.0))
        p5 = float(np.clip(probs_by_target[5][i], 0.0, 1.0))
        p10 = float(np.clip(probs_by_target[10][i], 0.0, 1.0))
        p20 = float(np.clip(probs_by_target[20][i], 0.0, 1.0))

        if p3 < min_prob_3:
            continue

        trade_style, style_reason = classify_trade_style(meta["last_row"], p3, p5, p10, p20)
        vol_bonus = 0.03 if meta["vol_ratio"] >= 1.5 else (-0.02 if meta["vol_ratio"] < 0.6 else 0.0)
        overheat_penalty = -0.04 if meta["rsi_14"] >= 72.0 else 0.0
        score = p3 * 0.35 + p5 * 0.30 + p10 * 0.20 + p20 * 0.15 + vol_bonus + overheat_penalty

        candidates.append({
            **meta,
            "trade_style": trade_style,
            "style_reason": style_reason,
            "prob_3": round(p3 * 100, 1),
            "prob_5": round(p5 * 100, 1),
            "prob_10": round(p10 * 100, 1),
            "prob_20": round(p20 * 100, 1),
            "score": round(score * 100, 2),
        })

    if not candidates:
        return pd.DataFrame()

    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = candidates[:top_n]

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []

    for item in selected:
        live_price = _fetch_live_1m_price(item["market"], item["current_price"])
        time.sleep(0.08)

        rec_buy, stop_loss, buy_window, micro_memo = analyze_micro_timing(
            market=item["market"],
            current_price=live_price,
            atr_pct=item["atr_pct"],
            rsi_14=item["rsi_14"],
            trade_style=item["trade_style"],
        )
        t3, t5, t10, t20 = _style_targets(rec_buy, item["trade_style"])
        combined_memo = f"{item['trade_style']}({item['style_reason']}) | {micro_memo}"

        row_out = {
            "예측시각": now_str,
            "대응유형": item["trade_style"],
            "코인": item["market"],
            "현재가": live_price,
            "추천매수가": rec_buy,
            "손절가": stop_loss,
            "목표1": t3,
            "목표2": t5,
            "목표3": t10,
            "목표4": t20,
            "+3%확률(%)": item["prob_3"],
            "+5%확률(%)": item["prob_5"],
            "+10%확률(%)": item["prob_10"],
            "+20%확률(%)": item["prob_20"],
            "예상매수시점": buy_window,
            "대응/1m5m메모": combined_memo,
            "종합점수": item["score"],
        }
        results.append(row_out)

        if save_to_db:
            _exec_write(
                """
                INSERT INTO prediction_logs (
                    predicted_at, market, trade_style, current_price, rec_buy_price, stop_loss,
                    target_3, target_5, target_10, target_20,
                    prob_3, prob_5, prob_10, prob_20, expected_buy_window,
                    status, outcome, max_gain_pct, hit_3, hit_5, hit_10, hit_20
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', 'PENDING', 0.0, 0, 0, 0, 0)
                """,
                (
                    now_str, item["market"], item["trade_style"], live_price, rec_buy, stop_loss,
                    t3, t5, t10, t20,
                    item["prob_3"], item["prob_5"], item["prob_10"], item["prob_20"],
                    buy_window,
                ),
            )

        time.sleep(0.1)

    return pd.DataFrame(results)


def scan_swing_tracks(
    top_n: int = 5,
    min_prob: float = 0.65,
    prob_col_target: int = 5,
    save_to_db: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    확률 컷(35~85% 조절 가능)을 유지하면서
    1. 추천 5~10% 안정스윙용
    2. 추천 10~20% 중기추세용
    두 그룹으로 나누어 반환합니다.
    """
    ensure_log_columns()
    models = load_trained_models()
    daily_map = load_daily_context_map()
    h4_map = load_h4_context_map()
    btc_map = load_btc_60m_map()
    markets = [m for m in get_krw_markets() if m not in STABLECOINS]

    conn = _connect_db(timeout=30.0)
    try:
        df_all = pd.read_sql_query(
            """
            SELECT market, candle_time_kst, open, high, low, close, volume
            FROM candles
            WHERE timeframe = '60m'
            ORDER BY market, candle_time_kst
            """,
            conn,
        )
    finally:
        conn.close()

    if df_all.empty:
        return {"안정스윙": pd.DataFrame(), "중기추세": pd.DataFrame()}

    grouped_60m = {str(m): grp for m, grp in df_all.groupby("market")}
    feature_batches = []
    meta_batches = []

    for market in markets:
        grp = grouped_60m.get(market)
        if grp is None or len(grp) < 130:
            continue
        d = enrich_dataframe(grp.sort_values("candle_time_kst"), market, daily_map, h4_map, btc_map)
        last_row = d.iloc[-1]
        if last_row[FEATURE_COLS].isna().any():
            continue
        if float(last_row["rsi_14"]) >= 80.0 or float(last_row["ret_3h"]) <= -0.08:
            continue

        feature_batches.append(last_row[FEATURE_COLS].astype(float).values)
        meta_batches.append({
            "market": market,
            "last_row": last_row,
            "current_price": float(last_row["close"]),
            "atr_pct": float(last_row["atr_pct"]),
            "rsi_14": float(last_row["rsi_14"]),
            "vol_ratio": float(last_row["vol_ratio"]),
        })

    if not feature_batches:
        return {"안정스윙": pd.DataFrame(), "중기추세": pd.DataFrame()}

    X_latest = np.vstack(feature_batches)
    probs_by_target = {pct: models[pct].predict(X_latest) for pct in TARGETS}

    stable_cands = []
    midtrend_cands = []

    for i, meta in enumerate(meta_batches):
        p3 = float(np.clip(probs_by_target[3][i], 0.0, 1.0))
        p5 = float(np.clip(probs_by_target[5][i], 0.0, 1.0))
        p10 = float(np.clip(probs_by_target[10][i], 0.0, 1.0))
        p20 = float(np.clip(probs_by_target[20][i], 0.0, 1.0))

        prob_map = {3: p3, 5: p5, 10: p10, 20: p20}
        filter_prob = prob_map.get(prob_col_target, p5)
        if filter_prob < min_prob:
            continue

        track, reason = classify_swing_track(meta["last_row"], p5, p10, p20)
        if track == "중기추세":
            score = p10 * 0.45 + p20 * 0.35 + p5 * 0.20
            midtrend_cands.append({
                **meta,
                "track": track,
                "reason": reason,
                "prob_3": round(p3 * 100, 1),
                "prob_5": round(p5 * 100, 1),
                "prob_10": round(p10 * 100, 1),
                "prob_20": round(p20 * 100, 1),
                "score": round(score * 100, 2),
            })
        else:
            score = p5 * 0.45 + p10 * 0.35 + p3 * 0.20
            stable_cands.append({
                **meta,
                "track": track,
                "reason": reason,
                "prob_3": round(p3 * 100, 1),
                "prob_5": round(p5 * 100, 1),
                "prob_10": round(p10 * 100, 1),
                "prob_20": round(p20 * 100, 1),
                "score": round(score * 100, 2),
            })

    stable_cands.sort(key=lambda x: x["score"], reverse=True)
    midtrend_cands.sort(key=lambda x: x["score"], reverse=True)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    grouped_results = {"안정스윙": [], "중기추세": []}

    for track_key, cand_list in [("안정스윙", stable_cands[:top_n]), ("중기추세", midtrend_cands[:top_n])]:
        for item in cand_list:
            live_price = _fetch_live_1m_price(item["market"], item["current_price"])
            time.sleep(0.08)

            rec_buy, stop_loss, buy_window, micro_memo = analyze_micro_timing(
                market=item["market"],
                current_price=live_price,
                atr_pct=item["atr_pct"],
                rsi_14=item["rsi_14"],
                trade_style="스윙",
            )
            t1, t2, t3, t4 = _style_targets(rec_buy, track_key)

            row_out = {
                "예측시각": now_str,
                "트랙": track_key,
                "코인": item["market"],
                "현재가": live_price,
                "추천매수가": rec_buy,
                "손절가": stop_loss,
                "목표1": t1,
                "목표2": t2,
                "목표3": t3,
                "목표4": t4,
                "+5%확률(%)": item["prob_5"],
                "+10%확률(%)": item["prob_10"],
                "+20%확률(%)": item["prob_20"],
                "예상매수시점": buy_window,
                "근거메모": f"{item['reason']} | {micro_memo}",
                "종합점수": item["score"],
            }
            grouped_results[track_key].append(row_out)

            if save_to_db:
                _exec_write(
                    """
                    INSERT INTO prediction_logs (
                        predicted_at, market, trade_style, current_price, rec_buy_price, stop_loss,
                        target_3, target_5, target_10, target_20,
                        prob_3, prob_5, prob_10, prob_20, expected_buy_window,
                        status, outcome, max_gain_pct, hit_3, hit_5, hit_10, hit_20
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', 'PENDING', 0.0, 0, 0, 0, 0)
                    """,
                    (
                        now_str, item["market"], track_key, live_price, rec_buy, stop_loss,
                        t1, t2, t3, t4,
                        item["prob_3"], item["prob_5"], item["prob_10"], item["prob_20"],
                        buy_window,
                    ),
                )
            time.sleep(0.08)

    return {
        "안정스윙": pd.DataFrame(grouped_results["안정스윙"]),
        "중기추세": pd.DataFrame(grouped_results["중기추세"]),
    }


def verify_predictions(use_1m_recent: bool = True, success_target_pct: int = 3) -> pd.DataFrame:
    ensure_log_columns()
    conn = _connect_db(timeout=30.0)
    try:
        logs = pd.read_sql_query(
            """
            SELECT * FROM prediction_logs
            WHERE outcome = 'PENDING'
               OR (outcome = 'SUCCESS' AND status NOT IN ('HIT_20'))
            ORDER BY id ASC
            """,
            conn,
        )
    finally:
        conn.close()

    if logs.empty:
        return pd.DataFrame()

    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    updates = []

    for _, row in logs.iterrows():
        market = row["market"]
        trade_style = str(row["trade_style"]) if pd.notna(row.get("trade_style")) else "단타"
        max_hold_hours = HOLD_HOURS_BY_STYLE.get(trade_style, 24.0)

        pred_dt = datetime.fromisoformat(row["predicted_at"])
        elapsed_hours = (now_dt - pred_dt).total_seconds() / 3600.0

        req_count = min(60, max(5, int(elapsed_hours) + 3))
        candles = fetch_candles(market, timeframe="60m", count=req_count)
        time.sleep(0.1)
        if not candles:
            continue

        after_candles = [
            c for c in candles
            if datetime.fromisoformat(c["candle_date_time_kst"]) >= pred_dt - timedelta(hours=1)
        ]
        if not after_candles:
            after_candles = candles[:3]

        min_low = min(float(c["low_price"]) for c in after_candles)
        max_high = max(float(c["high_price"]) for c in after_candles)

        if use_1m_recent and elapsed_hours <= 3.0:
            df_1m = fetch_realtime_candles(market, timeframe="1m", count=180)
            time.sleep(0.08)
            if not df_1m.empty:
                recent_1m = df_1m[df_1m["candle_time_kst"] >= row["predicted_at"]]
                if not recent_1m.empty:
                    min_low = min(min_low, float(recent_1m["low"].min()))
                    max_high = max(max_high, float(recent_1m["high"].max()))

        rec_buy = float(row["rec_buy_price"])
        stop_loss = float(row["stop_loss"])
        status = row["status"]
        outcome = row["outcome"] if pd.notna(row["outcome"]) else "PENDING"

        was_bought = status in ("BOUGHT", "HIT_3", "HIT_5", "HIT_10", "HIT_20") or (min_low <= rec_buy)
        if was_bought and status == "PENDING":
            status = "BOUGHT"

        base_ref = rec_buy if was_bought else float(row["current_price"])
        prev_max_gain = float(row["max_gain_pct"]) if pd.notna(row["max_gain_pct"]) else 0.0
        cur_gain_pct = ((max_high - base_ref) / base_ref * 100.0) if base_ref > 0 else 0.0
        max_gain_pct = round(max(prev_max_gain, cur_gain_pct), 2)

        hit_3 = int(bool(row["hit_3"]) or (was_bought and max_high >= float(row["target_3"])))
        hit_5 = int(bool(row["hit_5"]) or (was_bought and max_high >= float(row["target_5"])))
        hit_10 = int(bool(row["hit_10"]) or (was_bought and max_high >= float(row["target_10"])))
        hit_20 = int(bool(row["hit_20"]) or (was_bought and max_high >= float(row["target_20"])))

        if was_bought:
            if hit_20:
                status = "HIT_20"
            elif hit_10:
                status = "HIT_10"
            elif hit_5:
                status = "HIT_5"
            elif hit_3:
                status = "HIT_3"

        target_hit_map = {3: hit_3, 5: hit_5, 10: hit_10, 20: hit_20}
        if was_bought and target_hit_map.get(success_target_pct, hit_3):
            outcome = "SUCCESS"
        elif was_bought and min_low <= stop_loss and not hit_3:
            status = "STOPPED"
            outcome = "FAIL"
        elif elapsed_hours >= max_hold_hours:
            if not was_bought:
                status = "UNFILLED"
                outcome = "FAIL"
            elif not hit_3:
                status = "EXPIRED"
                outcome = "FAIL"
            else:
                outcome = "SUCCESS"

        if (
            status != row["status"]
            or outcome != row["outcome"]
            or max_gain_pct != float(row["max_gain_pct"] or 0.0)
        ):
            updates.append(
                (status, outcome, max_gain_pct, hit_3, hit_5, hit_10, hit_20, now_str, int(row["id"]))
            )

    if updates:
        _exec_write(
            lambda c: c.executemany(
                """
                UPDATE prediction_logs
                SET status = ?, outcome = ?, max_gain_pct = ?,
                    hit_3 = ?, hit_5 = ?, hit_10 = ?, hit_20 = ?, verified_at = ?
                WHERE id = ?
                """,
                updates,
            )
        )

    conn = _connect_db(timeout=30.0)
    try:
        summary = pd.read_sql_query(
            """
            SELECT outcome, status, COUNT(*) as cnt
            FROM prediction_logs
            GROUP BY outcome, status
            """,
            conn,
        )
    finally:
        conn.close()

    print(f"[{now_str}] 검증 완료: {len(updates)}건 상태 업데이트됨")
    return summary


def get_prediction_statistics() -> dict:
    ensure_log_columns()
    conn = _connect_db(timeout=30.0)
    try:
        df = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY id DESC", conn)
    finally:
        conn.close()

    if df.empty:
        return {"total": 0, "settled": 0, "success": 0, "fail": 0, "win_rate": 0.0, "target_stats": pd.DataFrame()}

    settled_df = df[df["outcome"].isin(["SUCCESS", "FAIL"])]
    bought_df = df[df["status"].isin(["BOUGHT", "HIT_3", "HIT_5", "HIT_10", "HIT_20", "STOPPED", "EXPIRED"])]

    total = len(df)
    settled = len(settled_df)
    success = int((settled_df["outcome"] == "SUCCESS").sum())
    fail = int((settled_df["outcome"] == "FAIL").sum())
    win_rate = round((success / settled * 100.0), 1) if settled > 0 else 0.0

    target_rows = []
    denom = len(bought_df) if len(bought_df) > 0 else len(settled_df)
    for pct in TARGETS:
        hit_col = f"hit_{pct}"
        prob_col = f"prob_{pct}"
        actual_hits = int(bought_df[hit_col].sum()) if len(bought_df) > 0 else 0
        actual_rate = round((actual_hits / denom * 100.0), 1) if denom > 0 else 0.0
        avg_pred_prob = round(float(df[prob_col].mean()), 1) if len(df) > 0 else 0.0
        target_rows.append({
            "목표": f"+{pct}%",
            "체결/검증모수": denom,
            "실제도달건수": actual_hits,
            "실제달성확률(%)": actual_rate,
            "AI평균예측확률(%)": avg_pred_prob,
            "차이(%p)": round(actual_rate - avg_pred_prob, 1),
        })

    return {
        "total": total,
        "settled": settled,
        "success": success,
        "fail": fail,
        "win_rate": win_rate,
        "target_stats": pd.DataFrame(target_rows),
    }


def scan_market_with_watchlist(
    top_n: int = 10,
    watchlist_n: int = 8,
    min_prob_3: float = 0.55,
    watch_min_prob_3: float = 0.45,
    save_to_db: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    1. recommended: min_prob_3 이상 실전 추천 후보(prediction_logs 기록 대상)
    2. watchlist: min_prob_3 미달이지만 추세/수급/과열 필터를 통과한 관심/대기 후보(로그 미기록)
    """
    recommended_df = scan_market(
        top_n=top_n,
        min_prob_3=min_prob_3,
        save_to_db=save_to_db,
    )

    models = load_trained_models()
    daily_map = load_daily_context_map()
    h4_map = load_h4_context_map()
    btc_map = load_btc_60m_map()
    markets = [m for m in get_krw_markets() if m not in STABLECOINS]

    conn = _connect_db(timeout=30.0)
    try:
        df_all = pd.read_sql_query(
            """
            SELECT market, candle_time_kst, open, high, low, close, volume
            FROM candles
            WHERE timeframe = '60m'
            ORDER BY market, candle_time_kst
            """,
            conn,
        )
    finally:
        conn.close()

    if df_all.empty:
        return {"recommended": recommended_df, "watchlist": pd.DataFrame()}

    rec_markets = set(recommended_df["코인"].tolist()) if not recommended_df.empty else set()
    grouped_60m = {str(m): grp for m, grp in df_all.groupby("market")}
    feature_batches = []
    meta_batches = []

    for market in markets:
        if market in rec_markets or market in STABLECOINS:
            continue
        grp = grouped_60m.get(market)
        if grp is None or len(grp) < 130:
            continue
        d = enrich_dataframe(grp.sort_values("candle_time_kst"), market, daily_map, h4_map, btc_map)
        last_row = d.iloc[-1]
        if last_row[FEATURE_COLS].isna().any():
            continue

        rsi_14 = float(last_row["rsi_14"])
        ret_3h = float(last_row["ret_3h"])
        vol_ratio = float(last_row["vol_ratio"])
        h4_ema20 = float(last_row["h4_ema20_dist"])
        d_ema20 = float(last_row["d_ema20_dist"])

        if rsi_14 >= 72.0 or ret_3h <= -0.08:
            continue
        if not (h4_ema20 >= -0.015 or d_ema20 >= -0.025 or vol_ratio >= 1.3):
            continue

        feature_batches.append(last_row[FEATURE_COLS].astype(float).values)
        meta_batches.append({
            "market": market,
            "last_row": last_row,
            "current_price": float(last_row["close"]),
            "atr_pct": float(last_row["atr_pct"]),
            "rsi_14": rsi_14,
            "vol_ratio": vol_ratio,
            "h4_ema20": h4_ema20,
            "d_ema20": d_ema20,
        })

    if not feature_batches:
        return {"recommended": recommended_df, "watchlist": pd.DataFrame()}

    X_latest = np.vstack(feature_batches)
    probs_by_target = {pct: models[pct].predict(X_latest) for pct in TARGETS}
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    watch_rows = []

    for i, meta in enumerate(meta_batches):
        p3 = float(np.clip(probs_by_target[3][i], 0.0, 1.0))
        if p3 < watch_min_prob_3 or p3 >= min_prob_3:
            continue

        p5 = float(np.clip(probs_by_target[5][i], 0.0, 1.0))
        p10 = float(np.clip(probs_by_target[10][i], 0.0, 1.0))
        p20 = float(np.clip(probs_by_target[20][i], 0.0, 1.0))
        trade_style, style_reason = classify_trade_style(meta["last_row"], p3, p5, p10, p20)

        score = p3 * 0.35 + p5 * 0.30 + p10 * 0.20 + p20 * 0.15
        watch_tags = []
        if meta["h4_ema20"] >= -0.015:
            watch_tags.append("4H 지지")
        if meta["d_ema20"] >= -0.025:
            watch_tags.append("일봉 지지")
        if meta["vol_ratio"] >= 1.3:
            watch_tags.append(f"거래량 {meta['vol_ratio']:.1f}배")

        watch_rows.append({
            "예측시각": now_str,
            "상태": "관심/대기",
            "대응유형": trade_style,
            "코인": meta["market"],
            "현재가": snap_to_upbit_tick(meta["current_price"]),
            "+3%확률(%)": round(p3 * 100, 1),
            "+5%확률(%)": round(p5 * 100, 1),
            "+10%확률(%)": round(p10 * 100, 1),
            "+20%확률(%)": round(p20 * 100, 1),
            "대기사유": ", ".join(watch_tags) if watch_tags else style_reason,
            "종합점수": round(score * 100, 2),
        })

    watch_rows.sort(key=lambda x: x["종합점수"], reverse=True)
    return {
        "recommended": recommended_df,
        "watchlist": pd.DataFrame(watch_rows[:watchlist_n]),
    }


if __name__ == "__main__":
    train_models()
    df_scan = scan_market(top_n=10, min_prob_3=0.55, save_to_db=True)
    if not df_scan.empty:
        print("\n=== 오늘의 추천 스캔 결과 ===")
        print(df_scan.to_string(index=False))
    df_ver = verify_predictions()
    if not df_ver.empty:
        print("\n=== 예측 로그 상태 집계 ===")
        print(df_ver.to_string(index=False))