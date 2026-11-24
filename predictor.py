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
HORIZON_CANDLES = 24          # 향후 24시간(60분봉 24개) 기준
STOP_LOSS_BY_TARGET = {
    3: 0.025,   # +3% 노릴 때 -2.5% 손절 기준
    5: 0.030,   # +5% 노릴 때 -3.0% 손절 기준
    10: 0.040,  # +10% 노릴 때 -4.0% 손절 기준
    20: 0.050,  # +20% 노릴 때 -5.0% 손절 기준
}


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
    """기존 prediction_logs 테이블에 통계용 컬럼이 없으면 안전하게 추가"""
    init_db()

    def _alter(conn):
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(prediction_logs)")
        existing_cols = {row[1] for row in cur.fetchall()}

        alter_stmts = [
            ("max_gain_pct", "REAL DEFAULT 0.0"),
            ("hit_3", "INTEGER DEFAULT 0"),
            ("hit_5", "INTEGER DEFAULT 0"),
            ("hit_10", "INTEGER DEFAULT 0"),
            ("hit_20", "INTEGER DEFAULT 0"),
            ("outcome", "TEXT DEFAULT 'PENDING'"),  # PENDING / SUCCESS / FAIL
        ]
        for col_name, col_def in alter_stmts:
            if col_name not in existing_cols:
                cur.execute(f"ALTER TABLE prediction_logs ADD COLUMN {col_name} {col_def}")

    _exec_write(_alter)


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
        hit_stop = False
        for j in range(i + 1, i + 1 + HORIZON_CANDLES):
            if lows[j] <= stop_price:
                hit_stop = True
                break
            if highs[j] >= target_price:
                hit_target = True
                break

        labels[i] = 1.0 if (hit_target and not hit_stop) else 0.0

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
    """스캔 시점의 최신 1분봉 종가를 반환하고, 조회 실패 시 fallback_price를 반환"""
    try:
        df_1m = fetch_realtime_candles(market, timeframe="1m", count=1)
        if df_1m is not None and not df_1m.empty and "close" in df_1m.columns:
            live_p = float(df_1m["close"].iloc[-1])
            if live_p > 0:
                return live_p
    except Exception:
        pass
    return float(fallback_price)


def analyze_micro_timing(market: str, current_price: float, atr_pct: float, rsi_14: float):
    df_15m = fetch_realtime_candles(market, timeframe="15m", count=96)
    time.sleep(0.08)
    df_5m = fetch_realtime_candles(market, timeframe="5m", count=60)
    time.sleep(0.08)
    df_1m = fetch_realtime_candles(market, timeframe="1m", count=30)

    safe_atr_pct = float(np.clip(atr_pct, 0.008, 0.05))
    notes = []

    if df_15m.empty or len(df_15m) < 20:
        rec_buy = current_price * (1 - 0.5 * safe_atr_pct)
        stop_loss = rec_buy * (1 - 1.5 * safe_atr_pct)
        return round(rec_buy, 4), round(stop_loss, 4), "즉시 ~ 30분 이내", "15m 데이터 부족(기본 ATR 기준)"

    ema20_15m = float(df_15m["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    recent_low_15m = float(df_15m["low"].tail(8).min())

    candidate_buy = min(current_price * (1 - 0.3 * safe_atr_pct), max(ema20_15m, recent_low_15m))
    rec_buy = float(np.clip(candidate_buy, current_price * 0.975, current_price))

    vol_5m_spike = False
    if not df_5m.empty and len(df_5m) >= 15:
        vol_ma_5m = df_5m["volume"].tail(15).mean()
        last_vol_5m = float(df_5m["volume"].iloc[-1])
        low_5m = float(df_5m["low"].tail(6).min())
        if low_5m < rec_buy:
            rec_buy = max(low_5m, current_price * 0.975)
        if vol_ma_5m > 0 and last_vol_5m >= vol_ma_5m * 1.8:
            vol_5m_spike = True
            notes.append("5m 거래량 급증")

    spike_1m = False
    if not df_1m.empty and len(df_1m) >= 6:
        ret_5m_from_1m = (float(df_1m["close"].iloc[-1]) - float(df_1m["close"].iloc[-6])) / float(df_1m["close"].iloc[-6])
        if abs(ret_5m_from_1m) >= 0.015:
            spike_1m = True
            notes.append(f"1m 직전 5분 변동 {ret_5m_from_1m * 100:+.1f}%")

    gap_pct = ((current_price - rec_buy) / current_price) * 100.0
    if rsi_14 >= 72 or gap_pct >= 1.2:
        expected_window = "약 1 ~ 3시간 내 (단기 과열/눌림목 대기)"
    elif spike_1m and gap_pct >= 0.3:
        expected_window = "약 15 ~ 45분 내 (1m 급등락 진정 후 진입)"
    elif gap_pct >= 0.4:
        expected_window = "약 30분 ~ 1시간 내 (눌림목 접근 중)"
    else:
        expected_window = "즉시 ~ 30분 이내 진입 유효" + (" (5m 수급 양호)" if vol_5m_spike else "")

    stop_loss = rec_buy * (1 - max(0.02, 1.5 * safe_atr_pct))
    detail_memo = " | ".join(notes) if notes else "안정 구간"
    return round(rec_buy, 4), round(stop_loss, 4), expected_window, detail_memo


def scan_market(top_n: int = 10, min_prob_3: float = 0.55, save_to_db: bool = True) -> pd.DataFrame:
    ensure_log_columns()
    models = load_trained_models()
    daily_map = load_daily_context_map()
    h4_map = load_h4_context_map()
    btc_map = load_btc_60m_map()
    markets = get_krw_markets()

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

    feature_batches = []
    meta_batches = []

    for market in markets:
        grp = df_all[df_all["market"] == market]
        if len(grp) < 130:
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
    fallback_list = []

    for i, meta in enumerate(meta_batches):
        p3 = float(np.clip(probs_by_target[3][i], 0.0, 1.0))
        p5 = float(np.clip(probs_by_target[5][i], 0.0, 1.0))
        p10 = float(np.clip(probs_by_target[10][i], 0.0, 1.0))
        p20 = float(np.clip(probs_by_target[20][i], 0.0, 1.0))

        vol_bonus = 0.03 if meta["vol_ratio"] >= 1.5 else (-0.02 if meta["vol_ratio"] < 0.6 else 0.0)
        overheat_penalty = -0.04 if meta["rsi_14"] >= 72.0 else 0.0
        score = p3 * 0.35 + p5 * 0.30 + p10 * 0.20 + p20 * 0.15 + vol_bonus + overheat_penalty

        item_data = {
            **meta,
            "prob_3": round(p3 * 100, 1),
            "prob_5": round(p5 * 100, 1),
            "prob_10": round(p10 * 100, 1),
            "prob_20": round(p20 * 100, 1),
            "score": round(score * 100, 2),
        }
        fallback_list.append(item_data)

        if p3 >= min_prob_3:
            candidates.append(item_data)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = candidates[:top_n]

    if not selected and fallback_list:
        fallback_list.sort(key=lambda x: x["score"], reverse=True)
        selected = fallback_list[: min(top_n, 5)]

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
        )
        t3 = round(rec_buy * 1.03, 4)
        t5 = round(rec_buy * 1.05, 4)
        t10 = round(rec_buy * 1.10, 4)
        t20 = round(rec_buy * 1.20, 4)

        row_out = {
            "예측시각": now_str,
            "코인": item["market"],
            "현재가": live_price,
            "추천매수가": rec_buy,
            "손절가": stop_loss,
            "목표+3%": t3,
            "목표+5%": t5,
            "목표+10%": t10,
            "목표+20%": t20,
            "+3%확률(%)": item["prob_3"],
            "+5%확률(%)": item["prob_5"],
            "+10%확률(%)": item["prob_10"],
            "+20%확률(%)": item["prob_20"],
            "예상매수시점": buy_window,
            "1m/5m메모": micro_memo,
            "종합점수": item["score"],
        }
        results.append(row_out)

        if save_to_db:
            _exec_write(
                """
                INSERT INTO prediction_logs (
                    predicted_at, market, current_price, rec_buy_price, stop_loss,
                    target_3, target_5, target_10, target_20,
                    prob_3, prob_5, prob_10, prob_20, expected_buy_window,
                    status, outcome, max_gain_pct, hit_3, hit_5, hit_10, hit_20
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', 'PENDING', 0.0, 0, 0, 0, 0)
                """,
                (
                    now_str, item["market"], live_price, rec_buy, stop_loss,
                    t3, t5, t10, t20,
                    item["prob_3"], item["prob_5"], item["prob_10"], item["prob_20"],
                    buy_window,
                ),
            )

        time.sleep(0.1)

    return pd.DataFrame(results)


def verify_predictions(use_1m_recent: bool = True, success_target_pct: int = 3) -> pd.DataFrame:
    ensure_log_columns()
    conn = _connect_db(timeout=30.0)
    try:
        logs = pd.read_sql_query(
            """
            SELECT * FROM prediction_logs
            WHERE outcome = 'PENDING' OR status IN ('PENDING', 'BOUGHT', 'HIT_3', 'HIT_5', 'HIT_10')
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
        pred_dt = datetime.fromisoformat(row["predicted_at"])
        elapsed_hours = (now_dt - pred_dt).total_seconds() / 3600.0

        candles = fetch_candles(market, timeframe="60m", count=min(48, max(5, int(elapsed_hours) + 3)))
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
        elif elapsed_hours >= HORIZON_CANDLES:
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