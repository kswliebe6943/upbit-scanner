from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import requests
import streamlit as st

import collector as upbit_collector
from collector import (
    DB_PATH,
    STABLECOINS,
    UPBIT_BASE_URL,
    get_krw_markets,
    init_db,
)
from predictor import (
    MODEL_DIR,
    TARGETS,
    ensure_log_columns,
    get_prediction_statistics,
    scan_market_with_watchlist,
    train_models,
    verify_predictions,
)

st.set_page_config(
    page_title="업비트 AI 코인 스캐너",
    page_icon="📈",
    layout="wide",
)


# =====================================================================
# 한국 시간(KST, UTC+9) 헬퍼
# =====================================================================
try:
    KST_TZ = ZoneInfo("Asia/Seoul")
except Exception:
    KST_TZ = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    """시간대 정보(tzinfo)가 제거된 순수 한국 시간(KST) naive datetime 반환"""
    return datetime.now(KST_TZ).replace(tzinfo=None)


def now_kst_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return now_kst().strftime(fmt)


# =====================================================================
# SQLite 동시 접근 잠금(database is locked) 방지 연결 헬퍼
# =====================================================================
_DB_WRITE_LOCK = threading.Lock()


def get_db_connection(timeout: float = 30.0) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=timeout, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except sqlite3.OperationalError:
        pass
    return conn


def _run_db_write(fn, max_retries: int = 8):
    last_exc = None
    for attempt in range(max_retries):
        try:
            with _DB_WRITE_LOCK:
                conn = get_db_connection(timeout=30.0)
                try:
                    res = fn(conn)
                    conn.commit()
                    return res
                finally:
                    conn.close()
        except sqlite3.OperationalError as exc:
            last_exc = exc
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                time.sleep(0.15 * (attempt + 1))
                continue
            raise
    if last_exc:
        raise last_exc


def safe_init_db():
    try:
        _run_db_write(lambda _conn: init_db())
    except Exception:
        init_db()


def safe_ensure_log_columns():
    try:
        _run_db_write(lambda _conn: ensure_log_columns())
    except Exception:
        ensure_log_columns()


safe_init_db()
safe_ensure_log_columns()


# =====================================================================
# 날짜 및 함수 호출 헬퍼
# =====================================================================
def _parse_dt_safe(val) -> datetime | None:
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().replace("T", " ")
    if "+" in s:
        s = s.split("+")[0].strip()
    if "Z" in s:
        s = s.replace("Z", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except Exception:
        return None


def _call_with_optional_progress(fn, *args, progress_cb=None, **kwargs):
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
    except Exception:
        params = {}

    for kw in ("progress_callback", "on_progress", "progress_cb", "callback"):
        if kw in params and kw not in kwargs and progress_cb is not None:
            kwargs[kw] = progress_cb
            break
    return fn(*args, **kwargs)


def _get_default_target_count(timeframe: str) -> int:
    tf_targets = getattr(upbit_collector, "TIMEFRAME_TARGETS", None)
    if isinstance(tf_targets, dict) and isinstance(tf_targets.get(timeframe), int) and tf_targets[timeframe] > 0:
        return tf_targets[timeframe]

    const_map = {
        "1d": ("TARGET_1D", "DAYS_COUNT", "DEFAULT_1D_COUNT", 365),
        "240m": ("TARGET_240M", "HOURS_4_COUNT", "DEFAULT_240M_COUNT", 2190),
        "60m": ("TARGET_60M", "HOURS_1_COUNT", "DEFAULT_60M_COUNT", 8760),
    }
    for attr_name in const_map.get(timeframe, ())[:-1]:
        val = getattr(upbit_collector, attr_name, None)
        if isinstance(val, int) and val > 0:
            return val
    return const_map.get(timeframe, (None, None, None, 365))[-1]


def run_collector_direct(progress_cb=None, stop_check=None, force: bool = False):
    def _stopped() -> bool:
        return bool(stop_check and stop_check())

    if _stopped():
        return

    if progress_cb:
        progress_cb(0.05, "최신 캔들 수집 준비 중")

    collect_fn = getattr(upbit_collector, "collect_timeframe_for_all", None)
    if callable(collect_fn):
        sig = inspect.signature(collect_fn)
        tf_plan = [
            ("1d", _get_default_target_count("1d"), 0.10, 0.40),
            ("240m", _get_default_target_count("240m"), 0.40, 0.72),
            ("60m", _get_default_target_count("60m"), 0.72, 0.98),
        ]
        for tf, cnt, start_r, end_r in tf_plan:
            if _stopped():
                break
            if progress_cb:
                progress_cb(start_r, f"[{tf}] 캔들 수집 중 (목표 {cnt}개)")
            kw = {}
            if "force" in sig.parameters:
                kw["force"] = force
            if "stop_check" in sig.parameters:
                kw["stop_check"] = stop_check
            _call_with_optional_progress(collect_fn, tf, cnt, progress_cb=progress_cb, **kw)
            if _stopped():
                break
            if progress_cb:
                progress_cb(end_r, f"[{tf}] 캔들 수집 완료")

        if not _stopped() and progress_cb:
            progress_cb(1.0, "1d / 240m / 60m 최신 캔들 수집 완료")
        return

    for fn_name in (
        "run_collection_pipeline",
        "collect_all_timeframes",
        "collect_all",
        "collect_candles",
        "collect_data",
        "update_candles",
        "update_all",
        "sync_candles",
        "run_collector",
        "main",
    ):
        fn = getattr(upbit_collector, fn_name, None)
        if callable(fn):
            if progress_cb:
                progress_cb(0.20, f"{fn_name}() 실행 중")
            _call_with_optional_progress(fn, progress_cb=progress_cb)
            if progress_cb:
                progress_cb(1.0, "캔들 수집 완료")
            return

    collector_file = Path(upbit_collector.__file__)
    if collector_file.exists():
        if progress_cb:
            progress_cb(0.20, "collector.py 직접 실행 중")
        proc = subprocess.run(
            [sys.executable, str(collector_file)],
            cwd=str(collector_file.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            err_tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            last_err = err_tail[-1] if err_tail else f"종료코드 {proc.returncode}"
            raise RuntimeError(f"collector.py 실행 실패: {last_err}")
        if progress_cb:
            progress_cb(1.0, "캔들 수집 완료")
        return

    raise AttributeError("collector.py에서 수집 실행 함수를 찾지 못했습니다.")


def run_training_direct(progress_cb=None):
    if progress_cb:
        progress_cb(0.15, "피처 생성 및 AI 모델 학습 중")
    res = _call_with_optional_progress(train_models, progress_cb=progress_cb)
    if progress_cb:
        progress_cb(1.0, "AI 모델 학습 완료")
    return res


def run_scan_direct(
    top_n: int,
    min_prob_3: float,
    watchlist_n: int = 8,
    watch_min_prob_3: float = 0.45,
    progress_cb=None,
) -> dict[str, pd.DataFrame]:
    if progress_cb:
        progress_cb(0.15, "전 종목 스캔 및 관심/대기 후보 분류 중")
    sig = inspect.signature(scan_market_with_watchlist)
    kwargs = {
        "top_n": top_n,
        "watchlist_n": watchlist_n,
        "min_prob_3": min_prob_3,
        "watch_min_prob_3": watch_min_prob_3,
        "save_to_db": True,
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    for kw in ("progress_callback", "on_progress", "progress_cb", "callback"):
        if kw in sig.parameters and progress_cb is not None:
            filtered_kwargs[kw] = progress_cb
            break
    res = scan_market_with_watchlist(**filtered_kwargs)
    if progress_cb:
        progress_cb(1.0, "전 종목 스캔 완료")
    if isinstance(res, dict):
        return {
            "recommended": res.get("recommended", pd.DataFrame()),
            "watchlist": res.get("watchlist", pd.DataFrame()),
        }
    return {"recommended": res if isinstance(res, pd.DataFrame) else pd.DataFrame(), "watchlist": pd.DataFrame()}


# =====================================================================
# 업비트 API 및 한글명/금액 포맷 헬퍼
# =====================================================================
def _fast_get_json(url: str, params: dict, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 429:
                time.sleep(0.30 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == max_retries - 1:
                return []
            time.sleep(0.15 * (attempt + 1))
    return []


def _fast_fetch_candles(market: str, timeframe: str = "5m", count: int = 25) -> list:
    params = {"market": market, "count": max(1, min(200, int(count)))}
    if timeframe == "1d":
        url = f"{UPBIT_BASE_URL}/candles/days"
    elif timeframe in {"1m", "5m", "15m", "60m", "240m"}:
        mins = timeframe.replace("m", "")
        url = f"{UPBIT_BASE_URL}/candles/minutes/{mins}"
    else:
        return []
    data = _fast_get_json(url, params)
    return data if isinstance(data, list) else []


@st.cache_data(ttl=3600)
def get_market_korean_name_map() -> dict:
    data = _fast_get_json(f"{UPBIT_BASE_URL}/market/all", {"isDetails": "false"})
    if not isinstance(data, list):
        return {}
    name_map = {}
    for item in data:
        m = item.get("market")
        kor = item.get("korean_name")
        if m and kor:
            name_map[str(m)] = str(kor)
    return name_map


def format_coin_label(market_val) -> str:
    if market_val is None or pd.isna(market_val):
        return "-"
    s = str(market_val).strip()
    if "(" in s and s.endswith(")"):
        return s
    name_map = get_market_korean_name_map()
    kor = name_map.get(s)
    if kor:
        return f"{s} ({kor})"
    return s


def attach_korean_coin_names(df: pd.DataFrame, col_name: str = "코인") -> pd.DataFrame:
    if df is None or df.empty or col_name not in df.columns:
        return df
    out = df.copy()
    out[col_name] = out[col_name].apply(format_coin_label)
    return out


def ensure_scalp_table():
    def _create(conn):
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scalp_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at TEXT NOT NULL,
                preset_name TEXT NOT NULL DEFAULT '커스텀',
                market TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                current_price REAL NOT NULL,
                rec_buy_price REAL NOT NULL,
                rec_sell_price REAL NOT NULL,
                target_pct REAL NOT NULL,
                stop_loss_pct REAL NOT NULL DEFAULT 2.0,
                verify_minutes INTEGER NOT NULL DEFAULT 15,
                timing TEXT NOT NULL,
                ret_pct REAL NOT NULL,
                vol_mult REAL NOT NULL,
                trade_val_krw REAL NOT NULL,
                bought INTEGER DEFAULT 0,
                max_gain_pct REAL DEFAULT 0.0,
                outcome TEXT NOT NULL DEFAULT 'PENDING',
                status_detail TEXT NOT NULL DEFAULT '대기중',
                verified_at TEXT
            )
            """
        )
        cols = [row[1] for row in cur.execute("PRAGMA table_info(scalp_logs)").fetchall()]
        if "preset_name" not in cols:
            cur.execute("ALTER TABLE scalp_logs ADD COLUMN preset_name TEXT NOT NULL DEFAULT '커스텀'")
        if "stop_loss_pct" not in cols:
            cur.execute("ALTER TABLE scalp_logs ADD COLUMN stop_loss_pct REAL NOT NULL DEFAULT 2.0")
        if "verify_minutes" not in cols:
            cur.execute("ALTER TABLE scalp_logs ADD COLUMN verify_minutes INTEGER NOT NULL DEFAULT 15")

    _run_db_write(_create)


ensure_scalp_table()


def reset_scalp_logs():
    ensure_scalp_table()
    _run_db_write(lambda conn: conn.execute("DELETE FROM scalp_logs"))


def reset_prediction_logs():
    safe_ensure_log_columns()
    _run_db_write(lambda conn: conn.execute("DELETE FROM prediction_logs"))


def get_krw_markets_by_volume(limit: int = 60) -> list:
    markets = get_krw_markets()
    if not markets:
        return []

    tickers = []
    chunk_size = 100
    for i in range(0, len(markets), chunk_size):
        batch = markets[i : i + chunk_size]
        data = _fast_get_json(f"{UPBIT_BASE_URL}/ticker", {"markets": ",".join(batch)})
        if isinstance(data, list):
            tickers.extend(data)

    if not tickers:
        return markets[:limit]

    vol_map = {
        item.get("market"): float(item.get("acc_trade_price_24h", 0.0))
        for item in tickers
        if item.get("market") and item.get("market") not in STABLECOINS
    }
    sorted_markets = sorted(
        [m for m in markets if m in vol_map],
        key=lambda m: vol_map[m],
        reverse=True,
    )
    return sorted_markets[:limit]


def format_price(val: float) -> float:
    if val >= 1000:
        return float(round(val, 0))
    if val >= 100:
        return float(round(val, 1))
    if val >= 10:
        return float(round(val, 2))
    if val >= 1:
        return float(round(val, 3))
    return float(round(val, 4))


def _to_float_safe(val):
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = str(val).replace(",", "").replace("원", "").strip()
    if cleaned == "" or cleaned == "-":
        return None
    return float(cleaned)


def fmt_krw(val) -> str:
    num = _to_float_safe(val)
    if num is None:
        return "-"
    if abs(num) >= 100:
        return f"{num:,.0f}"
    if abs(num) >= 10:
        return f"{num:,.2f}".rstrip("0").rstrip(".")
    return f"{num:,.4f}".rstrip("0").rstrip(".")


def fmt_int_comma(val) -> str:
    num = _to_float_safe(val)
    if num is None:
        return "-"
    return f"{int(round(num)):,}"


def format_price_columns(df: pd.DataFrame, price_cols: list, int_cols: list = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in price_cols:
        if col in out.columns:
            out[col] = out[col].apply(fmt_krw)
    if int_cols:
        for col in int_cols:
            if col in out.columns:
                out[col] = out[col].apply(fmt_int_comma)
    return out


def reorder_columns_front(df: pd.DataFrame, priority_cols: list) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    front = [c for c in priority_cols if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def _get_1m_pullback_price(market: str, cur_price: float, low_price: float) -> tuple[float | None, str | None]:
    """포착된 후보 종목에 한해 최근 1분봉 15개를 조회해 눌림 매수가 기준점을 계산"""
    raw_1m = _fast_fetch_candles(market, timeframe="1m", count=15)
    if not raw_1m or len(raw_1m) < 3:
        return None, None

    lows_1m = [float(c["low_price"]) for c in raw_1m[:10] if float(c.get("low_price", 0.0)) > 0]
    closes_1m = [float(c["trade_price"]) for c in raw_1m[:5] if float(c.get("trade_price", 0.0)) > 0]
    if not lows_1m:
        return None, None

    recent_1m_low = min(lows_1m)
    recent_1m_med_low = float(statistics.median(sorted(lows_1m)[: max(1, len(lows_1m) // 2)]))
    min_allowed = max(cur_price * 0.992, low_price * 0.998) if low_price > 0 else cur_price * 0.992

    if recent_1m_low < cur_price * 0.999:
        anchor = recent_1m_low if recent_1m_low >= min_allowed else recent_1m_med_low
        candidate = max(min_allowed, min(anchor, cur_price * 0.999))
        drop_pct = abs((candidate - cur_price) / cur_price * 100.0)
        return candidate, f"1분봉 눌림 확인 진입 (-{drop_pct:.2f}% 저점 부근)"

    if closes_1m and min(closes_1m) < cur_price * 0.9995:
        candidate = max(min_allowed, min(min(closes_1m) * 0.9995, cur_price * 0.999))
        drop_pct = abs((candidate - cur_price) / cur_price * 100.0)
        return candidate, f"1분봉 단기 조정 진입 (-{drop_pct:.2f}% 부근)"

    return None, None


def build_scalp_plan(
    cur_price: float,
    open_price: float,
    high_price: float,
    low_price: float,
    vol_mult: float,
    ret_pct: float,
    target_pct: float = 3.5,
    market: str | None = None,
):
    span = max(high_price - low_price, 0.0)
    upper_wick_ratio = ((high_price - max(cur_price, open_price)) / span) if span > 0 else 0.0

    pullback_buy, pullback_timing = (None, None)
    if market:
        time.sleep(0.02)
        pullback_buy, pullback_timing = _get_1m_pullback_price(market, cur_price, low_price)

    if pullback_buy is not None and pullback_timing is not None:
        buy_raw = pullback_buy
        timing = pullback_timing
    elif vol_mult >= 6.0 and ret_pct >= 4.5:
        buy_raw = cur_price * 0.9970
        timing = "초단기 숨고르기 후 진입 (3–10분 내 -0.3% 부근)"
    elif upper_wick_ratio >= 0.35 or ret_pct >= 3.2:
        buy_raw = cur_price * 0.9980
        timing = "단기 눌림 확인 진입 (2–8분 내 -0.2% 부근)"
    elif ret_pct >= 2.0:
        buy_raw = cur_price * 0.9990
        timing = "즉시–5분 내 눌림 진입 (-0.1% 부근)"
    else:
        buy_raw = cur_price * 0.9997
        timing = "즉시 진입 유효 (0–5분 내 현재가 근접)"

    lower_bound = max(low_price * 0.998, cur_price * 0.992) if low_price > 0 else cur_price * 0.992
    rec_buy = format_price(max(lower_bound, min(buy_raw, cur_price)))
    rec_sell = format_price(rec_buy * (1.0 + max(0.5, float(target_pct)) / 100.0))
    if rec_sell <= rec_buy:
        rec_sell = format_price(rec_buy * 1.01)
    return rec_buy, rec_sell, timing


def _refine_ai_buy_window(row_series: pd.Series) -> str:
    raw_win = str(row_series.get("expected_buy_window") or "").strip()
    cur_p = _to_float_safe(row_series.get("current_price"))
    buy_p = _to_float_safe(row_series.get("rec_buy_price"))

    if cur_p and buy_p and cur_p > 0:
        diff_pct = (buy_p - cur_p) / cur_p * 100.0
        if diff_pct >= -0.15:
            return "즉시–10분 내 진입 (현재가 근접)"
        if diff_pct >= -0.5:
            return "10–25분 내 눌림 진입 (-0.2%–-0.5%)"
        if diff_pct >= -1.2:
            return "25–45분 내 조정 진입 (-0.5%–-1.2%)"
        return "45–90분 내 깊은 눌림 대기 (-1.2% 초과)"

    if raw_win and raw_win not in ("-", "None", "nan", "1~3시간", "1~3시간 내"):
        return raw_win
    return "15–30분 내 눌림 확인 후 진입"


def load_db_stats() -> pd.DataFrame:
    conn = get_db_connection(timeout=30.0)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                timeframe,
                COUNT(DISTINCT market) AS coin_count,
                COUNT(*) AS candle_count,
                MIN(candle_time_kst) AS earliest_kst,
                MAX(candle_time_kst) AS latest_kst
            FROM candles
            GROUP BY timeframe
            ORDER BY timeframe
            """,
            conn,
        )
        return df
    finally:
        conn.close()


def _translate_outcome(val: str) -> str:
    return {
        "SUCCESS": "성공",
        "FAIL": "실패",
        "PENDING": "대기중",
    }.get(str(val), str(val))


def _translate_status(val: str) -> str:
    return {
        "PENDING": "대기중",
        "BOUGHT": "매수체결",
        "HIT_3": "+3% 도달",
        "HIT_5": "+5% 도달",
        "HIT_10": "+10% 도달",
        "HIT_20": "+20% 도달",
        "STOPPED": "손절이탈",
        "UNFILLED": "미체결종료",
        "EXPIRED": "기간만료(미달)",
    }.get(str(val), str(val))


def _classify_ai_reason(row) -> str:
    outcome = str(row.get("outcome") or "").upper()
    status = str(row.get("status") or "").upper()
    detail = str(row.get("status_detail") or "").strip() if "status_detail" in row else ""

    if outcome == "PENDING":
        return "대기중 (판정 전)"

    combined = f"{status} {detail}".upper()
    if outcome == "SUCCESS":
        if "HIT_20" in combined or "+20%" in combined or "20%" in combined:
            return "성공: 목표(+20%) 도달"
        if "HIT_10" in combined or "+10%" in combined or "10%" in combined:
            return "성공: 목표(+10%) 도달"
        if "HIT_5" in combined or "+5%" in combined or "5%" in combined:
            return "성공: 목표(+5%) 도달"
        if "HIT_3" in combined or "+3%" in combined or "3%" in combined:
            return "성공: 목표(+3%) 도달"
        return f"성공: {detail}" if detail else "성공: 목표가 도달"

    if "STOP" in combined or "손절" in combined:
        return "실패: 손절가 이탈"
    if "UNFILL" in combined or "미체결" in combined:
        return "실패: 추천매수가 미체결"
    if "EXPIRE" in combined or "만료" in combined or "미달" in combined:
        return "실패: 기간 내 목표가 미달"
    return f"실패: {detail}" if detail else "실패: 목표 미달/조건 미충족"


def get_ai_reason_summary() -> pd.DataFrame:
    safe_ensure_log_columns()
    conn = get_db_connection(timeout=30.0)
    try:
        df = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY id DESC", conn)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    df["_reason"] = df.apply(_classify_ai_reason, axis=1)
    settled = df[df["outcome"].isin(["SUCCESS", "FAIL"])]
    if settled.empty:
        return pd.DataFrame()

    grp = (
        settled.groupby("_reason", as_index=False)
        .size()
        .rename(columns={"_reason": "판정사유", "size": "건수"})
        .sort_values(by="건수", ascending=False)
        .reset_index(drop=True)
    )
    total_settled = len(settled)
    grp["비중(%)"] = grp["건수"].apply(lambda c: round(c / total_settled * 100.0, 1))
    grp["건수"] = grp["건수"].apply(fmt_int_comma)
    return grp


def load_prediction_logs(limit: int = 200, outcome_filter: str = "전체") -> pd.DataFrame:
    conn = get_db_connection(timeout=30.0)
    try:
        base_sql = """
            SELECT
                id, predicted_at, market, current_price, rec_buy_price, stop_loss,
                target_3, target_5, target_10, target_20,
                prob_3, prob_5, prob_10, prob_20,
                max_gain_pct, status, outcome, expected_buy_window, verified_at
            FROM prediction_logs
        """
        code_map = {"성공": "SUCCESS", "실패": "FAIL", "대기중": "PENDING"}
        if outcome_filter in code_map:
            df = pd.read_sql_query(
                base_sql + " WHERE outcome = ? ORDER BY id DESC LIMIT ?",
                conn,
                params=(code_map[outcome_filter], limit),
            )
        else:
            df = pd.read_sql_query(
                base_sql + " ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )
    finally:
        conn.close()

    if df.empty:
        return df

    df["판정사유"] = df.apply(_classify_ai_reason, axis=1)
    df["expected_buy_window"] = df.apply(_refine_ai_buy_window, axis=1)
    df["status"] = df["status"].apply(_translate_status)
    df["outcome"] = df["outcome"].apply(_translate_outcome)

    renamed = df.rename(columns={
        "id": "번호",
        "predicted_at": "예측(포착)시각",
        "market": "코인",
        "current_price": "포착시점현재가",
        "rec_buy_price": "추천매수가",
        "stop_loss": "손절가",
        "target_3": "목표+3%",
        "target_5": "목표+5%",
        "target_10": "목표+10%",
        "target_20": "목표+20%",
        "prob_3": "+3%확률(%)",
        "prob_5": "+5%확률(%)",
        "prob_10": "+10%확률(%)",
        "prob_20": "+20%확률(%)",
        "max_gain_pct": "최고수익(%)",
        "status": "진행상태",
        "outcome": "결과",
        "expected_buy_window": "예상매수시점",
        "verified_at": "검증시각",
    })
    renamed = attach_korean_coin_names(renamed, col_name="코인")
    return format_price_columns(
        renamed,
        price_cols=["포착시점현재가", "추천매수가", "손절가", "목표+3%", "목표+5%", "목표+10%", "목표+20%"],
        int_cols=["번호"],
    )


def models_ready() -> bool:
    return all((MODEL_DIR / f"lgb_target_{pct}.txt").exists() for pct in TARGETS)


# =====================================================================
# 단타 포착(초입 수급 필터 + 1분봉 눌림 보정 + 프리셋/검증기준 기록) 및 검증
# =====================================================================
def _evaluate_single_spike(
    m: str,
    timeframe: str,
    min_vol_mult: float,
    min_ret_pct: float,
    max_ret_pct: float,
    min_trade_krw: int,
    target_pct: float,
    stop_loss_pct: float = 2.0,
    verify_minutes: int = 15,
    preset_name: str = "커스텀",
):
    time.sleep(0.03)
    raw = _fast_fetch_candles(m, timeframe=timeframe, count=24)
    if not raw or len(raw) < 7:
        return None

    latest = raw[0]
    prev = raw[1]

    hist_vols = [
        float(c.get("candle_acc_trade_volume", c.get("candle_acc_volume", 0.0)))
        for c in raw[1:21]
        if float(c.get("candle_acc_trade_volume", c.get("candle_acc_volume", 0.0))) > 0
    ]
    if not hist_vols:
        return None

    median_vol = float(statistics.median(hist_vols))
    cur_vol = float(latest.get("candle_acc_trade_volume", latest.get("candle_acc_volume", 0.0)))
    vol_mult = (cur_vol / median_vol) if median_vol > 0 else 0.0

    cur_price = float(latest["trade_price"])
    prev_close = float(prev["trade_price"])
    open_price = float(latest["opening_price"])
    high_price = float(latest["high_price"])
    low_price = float(latest["low_price"])

    if cur_price <= 0 or open_price <= 0 or prev_close <= 0:
        return None

    ret_vs_prev_pct = ((cur_price - prev_close) / prev_close) * 100.0
    ret_candle_pct = ((cur_price - open_price) / open_price) * 100.0
    best_ret_pct = max(ret_vs_prev_pct, ret_candle_pct)
    trade_val_krw = float(latest.get("candle_acc_trade_price", cur_price * cur_vol))

    base_3_idx = min(3, len(raw) - 1)
    base_3_price = float(raw[base_3_idx]["trade_price"])
    cum_3_ret_pct = ((cur_price - base_3_price) / base_3_price) * 100.0 if base_3_price > 0 else 0.0

    candle_span = max(high_price - low_price, 0.0)
    upper_wick_ratio = ((high_price - max(cur_price, open_price)) / candle_span) if candle_span > 0 else 0.0

    if not (
        cur_price > open_price
        and vol_mult >= min_vol_mult
        and min_ret_pct <= best_ret_pct <= max_ret_pct
        and trade_val_krw >= min_trade_krw
        and upper_wick_ratio < 0.45
        and cum_3_ret_pct <= 8.5
    ):
        return None

    rec_buy, rec_sell, timing = build_scalp_plan(
        cur_price=cur_price,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        vol_mult=vol_mult,
        ret_pct=best_ret_pct,
        target_pct=target_pct,
        market=m,
    )
    cur_fmt = format_price(cur_price)
    ret_fmt = round(best_ret_pct, 2)
    vol_fmt = round(vol_mult, 2)
    val_man = int(trade_val_krw // 10_000)
    stop_fmt = round(float(stop_loss_pct), 1)
    ver_min_int = max(1, int(round(verify_minutes)))

    display_row = {
        "코인": format_coin_label(m),
        "세팅(프리셋)": preset_name,
        "검증기준(분)": ver_min_int,
        "손절기준(%)": stop_fmt,
        "추천매수금액": rec_buy,
        "추천매도금액": rec_sell,
        "거래량배율(배)": vol_fmt,
        "상승률(%)": ret_fmt,
        "추천매수타이밍": timing,
        "포착시점현재가": cur_fmt,
        "거래대금(만원)": val_man,
        "봉": timeframe,
    }
    db_tuple = (
        preset_name, m, timeframe, cur_fmt,
        rec_buy, rec_sell, float(target_pct), stop_fmt, ver_min_int, timing,
        ret_fmt, vol_fmt, trade_val_krw,
    )
    return display_row, db_tuple


def detect_volume_spikes(
    timeframe: str = "5m",
    min_vol_mult: float = 5.0,
    min_ret_pct: float = 2.0,
    max_ret_pct: float = 8.0,
    min_trade_krw: int = 30_000_000,
    target_pct: float = 3.5,
    stop_loss_pct: float = 2.0,
    verify_minutes: int = 15,
    max_coins: int = 60,
    preset_name: str = "커스텀",
    save_to_db: bool = True,
    progress_cb=None,
) -> pd.DataFrame:
    ensure_scalp_table()
    if progress_cb:
        progress_cb(0.10, "거래대금 상위 마켓 조회 중")
    markets = get_krw_markets_by_volume(limit=max_coins)
    if not markets:
        if progress_cb:
            progress_cb(1.0, "대상 마켓 없음")
        return pd.DataFrame()

    rows = []
    db_rows = []
    kst_now_str = now_kst_str("%Y-%m-%d %H:%M:%S")
    total_m = max(1, len(markets))
    done_m = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                _evaluate_single_spike,
                m,
                timeframe,
                min_vol_mult,
                min_ret_pct,
                max_ret_pct,
                min_trade_krw,
                target_pct,
                stop_loss_pct,
                verify_minutes,
                preset_name,
            )
            for m in markets
        ]
        for fut in as_completed(futures):
            done_m += 1
            if progress_cb:
                ratio = 0.10 + 0.75 * (done_m / total_m)
                progress_cb(ratio, f"단타 종목 분석 중 ({done_m}/{total_m})")
            res = fut.result()
            if res is not None:
                disp, tup = res
                rows.append(disp)
                db_rows.append((kst_now_str, *tup))

    if save_to_db and db_rows:
        if progress_cb:
            progress_cb(0.90, "단타 포착 로그 DB 저장 중")

        def _insert_scalp(conn):
            conn.executemany(
                """
                INSERT INTO scalp_logs (
                    detected_at, preset_name, market, timeframe, current_price,
                    rec_buy_price, rec_sell_price, target_pct, stop_loss_pct, verify_minutes, timing,
                    ret_pct, vol_mult, trade_val_krw,
                    bought, max_gain_pct, outcome, status_detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0.0, 'PENDING', '대기중')
                """,
                db_rows,
            )

        _run_db_write(_insert_scalp)

    if progress_cb:
        progress_cb(1.0, "단타 포착 완료")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["거래량배율(배)", "상승률(%)"], ascending=False).reset_index(drop=True)
    return df


def _filter_window_candles(raw_candles: list, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    if not raw_candles:
        return pd.DataFrame()
    df = pd.DataFrame(raw_candles)
    if "candle_date_time_kst" not in df.columns:
        return pd.DataFrame()

    df["_dt"] = df["candle_date_time_kst"].apply(_parse_dt_safe)
    df = df.dropna(subset=["_dt"])
    if df.empty:
        return pd.DataFrame()

    window_df = df[(df["_dt"] >= (start_dt - timedelta(minutes=1))) & (df["_dt"] <= end_dt)].copy()
    return window_df.sort_values("_dt")


def _verify_single_scalp(
    row_dict: dict,
    fallback_expire_hours: float,
    fallback_stop_loss_pct: float,
    now_dt: datetime,
):
    det_dt = _parse_dt_safe(row_dict.get("detected_at"))
    if det_dt is None:
        return None

    saved_ver_min = _to_float_safe(row_dict.get("verify_minutes"))
    if saved_ver_min is not None and saved_ver_min > 0:
        window_minutes = max(1, int(round(saved_ver_min)))
    else:
        window_minutes = max(1, int(round(fallback_expire_hours * 60)))

    saved_stop_pct = _to_float_safe(row_dict.get("stop_loss_pct"))
    row_stop_loss_pct = float(saved_stop_pct) if saved_stop_pct is not None and saved_stop_pct >= 0 else float(fallback_stop_loss_pct)

    market = row_dict["market"]
    end_dt = det_dt + timedelta(minutes=window_minutes)
    elapsed_minutes = (now_dt - det_dt).total_seconds() / 60.0

    needed_1m = int(min(180, max(10, window_minutes + 5)))
    raw_1m = _fast_fetch_candles(market, timeframe="1m", count=needed_1m)
    time.sleep(0.03)
    needed_5m = int(min(60, max(20, (window_minutes // 5) + 8)))
    raw_5m = _fast_fetch_candles(market, timeframe="5m", count=needed_5m)

    w_1m = _filter_window_candles(raw_1m, det_dt, min(now_dt, end_dt))
    w_5m = _filter_window_candles(raw_5m, det_dt, min(now_dt, end_dt))

    lows, highs = [], []
    if not w_1m.empty:
        lows.append(float(w_1m["low_price"].min()))
        highs.append(float(w_1m["high_price"].max()))
    if not w_5m.empty:
        lows.append(float(w_5m["low_price"].min()))
        highs.append(float(w_5m["high_price"].max()))

    if not lows or not highs:
        if elapsed_minutes < window_minutes:
            return None
        min_low = float(row_dict["current_price"])
        max_high = float(row_dict["current_price"])
    else:
        min_low = min(lows)
        max_high = max(highs)

    rec_buy = float(row_dict["rec_buy_price"])
    rec_sell = float(row_dict["rec_sell_price"])
    stop_price = rec_buy * (1.0 - max(0.0, row_stop_loss_pct) / 100.0)
    bought = int(bool(row_dict["bought"]) or (min_low <= rec_buy))

    base_ref = rec_buy if bought else float(row_dict["current_price"])
    cur_gain_pct = ((max_high - base_ref) / base_ref * 100.0) if base_ref > 0 else 0.0
    max_gain_pct = round(max(float(row_dict["max_gain_pct"] or 0.0), cur_gain_pct), 2)

    outcome = row_dict["outcome"]
    status_detail = row_dict["status_detail"]

    if bought and row_stop_loss_pct > 0 and min_low <= stop_price:
        outcome = "FAIL"
        status_detail = f"{window_minutes}분 기준 손절가(-{row_stop_loss_pct:.1f}% / {fmt_krw(stop_price)}) 이탈"
    elif bought and max_high >= rec_sell:
        outcome = "SUCCESS"
        status_detail = f"{window_minutes}분 기준 매수({fmt_krw(rec_buy)})·목표({fmt_krw(rec_sell)}) 도달"
    elif elapsed_minutes >= window_minutes:
        outcome = "FAIL"
        status_detail = (
            f"{window_minutes}분 기준 목표가 미달 (최저 {fmt_krw(min_low)} / 최고 {fmt_krw(max_high)})"
            if bought
            else f"{window_minutes}분 기준 매수가({fmt_krw(rec_buy)}) 미도달 (최저 {fmt_krw(min_low)})"
        )
    else:
        remain_min = max(1, int(round(window_minutes - elapsed_minutes)))
        status_detail = (
            f"대기중 · {window_minutes}분 기준 (매수체결 · 남은 {remain_min}분)"
            if bought
            else f"대기중 · {window_minutes}분 기준 (매수가 대기 · 남은 {remain_min}분)"
        )

    changed = (
        bought != int(row_dict["bought"] or 0)
        or max_gain_pct != float(row_dict["max_gain_pct"] or 0.0)
        or outcome != row_dict["outcome"]
        or status_detail != row_dict["status_detail"]
    )
    if not changed:
        return None

    return (bought, max_gain_pct, outcome, status_detail, int(row_dict["id"]))


def verify_scalp_predictions(
    expire_hours: float = 0.25,
    stop_loss_pct: float = 2.0,
    progress_cb=None,
) -> pd.DataFrame:
    ensure_scalp_table()
    if progress_cb:
        progress_cb(0.10, "단타 대기 로그 조회 중")
    conn = get_db_connection(timeout=30.0)
    try:
        logs = pd.read_sql_query(
            """
            SELECT * FROM scalp_logs
            WHERE outcome = 'PENDING'
            ORDER BY id ASC
            """,
            conn,
        )
    finally:
        conn.close()

    if logs.empty:
        if progress_cb:
            progress_cb(1.0, "검증할 대기 건 없음")
        return pd.DataFrame()

    kst_now_dt = now_kst()
    kst_now_str = kst_now_dt.strftime("%Y-%m-%d %H:%M:%S")
    updates = []
    total_l = max(1, len(logs))
    done_l = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                _verify_single_scalp,
                r.to_dict(),
                expire_hours,
                stop_loss_pct,
                kst_now_dt,
            )
            for _, r in logs.iterrows()
        ]
        for fut in as_completed(futures):
            done_l += 1
            if progress_cb:
                ratio = 0.10 + 0.80 * (done_l / total_l)
                progress_cb(ratio, f"단타 로그 개별 기준 채점 중 ({done_l}/{total_l})")
            res = fut.result()
            if res is not None:
                bought, max_gain_pct, outcome, status_detail, row_id = res
                updates.append((bought, max_gain_pct, outcome, status_detail, kst_now_str, row_id))

    if updates:
        def _apply_updates(conn):
            conn.executemany(
                """
                UPDATE scalp_logs
                SET bought = ?, max_gain_pct = ?, outcome = ?, status_detail = ?, verified_at = ?
                WHERE id = ?
                """,
                updates,
            )

        _run_db_write(_apply_updates)

    conn = get_db_connection(timeout=30.0)
    try:
        summary = pd.read_sql_query(
            "SELECT outcome, COUNT(*) AS cnt FROM scalp_logs GROUP BY outcome",
            conn,
        )
    finally:
        conn.close()

    if progress_cb:
        progress_cb(1.0, f"단타 검증 완료 ({len(updates)}건 갱신)")
    return summary


def get_scalp_statistics() -> dict:
    ensure_scalp_table()
    conn = get_db_connection(timeout=30.0)
    try:
        df = pd.read_sql_query("SELECT * FROM scalp_logs ORDER BY id DESC", conn)
    finally:
        conn.close()

    if df.empty:
        return {
            "total": 0,
            "settled": 0,
            "success": 0,
            "fail": 0,
            "win_rate": 0.0,
            "avg_max_gain": 0.0,
            "by_preset": pd.DataFrame(),
            "by_vol": pd.DataFrame(),
        }

    if "preset_name" not in df.columns:
        df["preset_name"] = "커스텀"
    df["preset_name"] = df["preset_name"].fillna("커스텀").replace({"": "커스텀"})

    settled_df = df[df["outcome"].isin(["SUCCESS", "FAIL"])]
    total = len(df)
    settled = len(settled_df)
    success = int((settled_df["outcome"] == "SUCCESS").sum())
    fail = int((settled_df["outcome"] == "FAIL").sum())
    win_rate = round((success / settled * 100.0), 1) if settled > 0 else 0.0
    avg_max_gain = round(float(settled_df["max_gain_pct"].mean()), 2) if settled > 0 else 0.0

    preset_rows = []
    ordered_presets = list(SCALP_PRESETS.keys()) + ["커스텀"]
    extra_presets = [p for p in df["preset_name"].unique().tolist() if p not in ordered_presets]
    for p_name in ordered_presets + extra_presets:
        sub_all = df[df["preset_name"] == p_name]
        if sub_all.empty:
            continue
        sub_set = settled_df[settled_df["preset_name"] == p_name]
        s_cnt = int((sub_set["outcome"] == "SUCCESS").sum())
        tot_s = len(sub_set)
        rate = round((s_cnt / tot_s * 100.0), 1) if tot_s > 0 else 0.0
        avg_g = round(float(sub_set["max_gain_pct"].mean()), 2) if tot_s > 0 else 0.0
        preset_rows.append({
            "세팅(프리셋)": p_name,
            "포착건수": fmt_int_comma(len(sub_all)),
            "판정건수": fmt_int_comma(tot_s),
            "성공건수": fmt_int_comma(s_cnt),
            "실제성공률(%)": rate,
            "평균최고수익(%)": avg_g,
        })

    vol_rows = []
    buckets = [
        ("2.5배 – 3.5배", 2.5, 3.5),
        ("3.5배 – 5.0배", 3.5, 5.0),
        ("5.0배 이상", 5.0, 999.0),
    ]
    for label, lo, hi in buckets:
        sub_all = df[(df["vol_mult"] >= lo) & (df["vol_mult"] < hi)]
        sub_set = settled_df[(settled_df["vol_mult"] >= lo) & (settled_df["vol_mult"] < hi)]
        s_cnt = int((sub_set["outcome"] == "SUCCESS").sum())
        tot_s = len(sub_set)
        rate = round((s_cnt / tot_s * 100.0), 1) if tot_s > 0 else 0.0
        vol_rows.append({
            "거래량배율구간": label,
            "포착건수": fmt_int_comma(len(sub_all)),
            "판정건수": fmt_int_comma(tot_s),
            "성공건수": fmt_int_comma(s_cnt),
            "실제성공률(%)": rate,
        })

    return {
        "total": total,
        "settled": settled,
        "success": success,
        "fail": fail,
        "win_rate": win_rate,
        "avg_max_gain": avg_max_gain,
        "by_preset": pd.DataFrame(preset_rows),
        "by_vol": pd.DataFrame(vol_rows),
    }


def load_scalp_logs(
    limit: int = 200,
    outcome_filter: str = "전체",
    preset_filter: str = "전체",
) -> pd.DataFrame:
    ensure_scalp_table()
    conn = get_db_connection(timeout=30.0)
    try:
        base_sql = """
            SELECT
                id, detected_at, preset_name, market, timeframe, current_price,
                rec_buy_price, rec_sell_price, target_pct, stop_loss_pct, verify_minutes, timing,
                ret_pct, vol_mult, bought, max_gain_pct,
                outcome, status_detail, verified_at
            FROM scalp_logs
        """
        conditions = []
        params = []
        code_map = {"성공": "SUCCESS", "실패": "FAIL", "대기중": "PENDING"}
        if outcome_filter in code_map:
            conditions.append("outcome = ?")
            params.append(code_map[outcome_filter])
        if preset_filter and preset_filter != "전체":
            conditions.append("preset_name = ?")
            params.append(preset_filter)

        where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        df = pd.read_sql_query(
            base_sql + where_clause + " ORDER BY id DESC LIMIT ?",
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["preset_name"] = df["preset_name"].fillna("커스텀").replace({"": "커스텀"})
    df["verify_minutes"] = df["verify_minutes"].fillna(15).astype(int)
    df["stop_loss_pct"] = df["stop_loss_pct"].fillna(2.0).astype(float).round(1)
    df["bought"] = df["bought"].apply(lambda x: "체결됨" if int(x or 0) == 1 else "미체결")
    df["outcome"] = df["outcome"].apply(_translate_outcome)
    df["status_detail"] = df["status_detail"].replace({"PENDING": "대기중"})

    renamed = df.rename(columns={
        "id": "번호",
        "detected_at": "포착시각",
        "preset_name": "세팅(프리셋)",
        "verify_minutes": "검증기준(분)",
        "stop_loss_pct": "손절기준(%)",
        "market": "코인",
        "timeframe": "봉",
        "current_price": "포착시점현재가",
        "rec_buy_price": "추천매수금액",
        "rec_sell_price": "추천매도금액",
        "target_pct": "목표(%)",
        "timing": "추천매수타이밍",
        "ret_pct": "상승률(%)",
        "vol_mult": "거래량배율(배)",
        "bought": "매수여부",
        "max_gain_pct": "최고수익(%)",
        "outcome": "결과",
        "status_detail": "상세상태",
        "verified_at": "검증시각",
    })
    renamed = attach_korean_coin_names(renamed, col_name="코인")
    formatted = format_price_columns(
        renamed,
        price_cols=["포착시점현재가", "추천매수금액", "추천매도금액"],
        int_cols=["번호", "검증기준(분)"],
    )
    return reorder_columns_front(
        formatted,
        [
            "포착시각",
            "세팅(프리셋)",
            "검증기준(분)",
            "손절기준(%)",
            "코인",
            "추천매수금액",
            "추천매도금액",
            "거래량배율(배)",
            "상승률(%)",
            "최고수익(%)",
            "매수여부",
            "결과",
            "상세상태",
        ],
    )


def _make_ui_progress_cb(bar, caption_box):
    def _cb(ratio: float, msg: str = ""):
        clamped = max(0.0, min(1.0, float(ratio)))
        pct = int(round(clamped * 100))
        bar.progress(clamped)
        caption_box.caption(f"⏳ {msg} ({pct}%)" if msg else f"⏳ 진행 중 ({pct}%)")
    return _cb


# =====================================================================
# 추천 단타 세팅 프리셋 정의(총 6종) 및 즉시 반영 콜백
# =====================================================================
SCALP_PRESETS = {
    "추천1. 5분 초단타 셋팅": {
        "spike_tf": "5m",
        "min_vol_mult": 5.0,
        "min_ret_pct": 2.0,
        "target_pct": 3.5,
        "stop_loss_pct": 2.0,
        "verify_minutes": 15,
        "max_coins": 60,
    },
    "추천2. 15분 단타 셋팅": {
        "spike_tf": "5m",
        "min_vol_mult": 3.5,
        "min_ret_pct": 1.5,
        "target_pct": 3.0,
        "stop_loss_pct": 2.0,
        "verify_minutes": 30,
        "max_coins": 80,
    },
    "추천3. 초입 저점 근사치 셋팅": {
        "spike_tf": "5m",
        "min_vol_mult": 4.0,
        "min_ret_pct": 1.0,
        "target_pct": 3.5,
        "stop_loss_pct": 2.2,
        "verify_minutes": 20,
        "max_coins": 120,
    },
    "추천4. 폭발 수급 슈팅형": {
        "spike_tf": "5m",
        "min_vol_mult": 7.0,
        "min_ret_pct": 2.5,
        "target_pct": 4.0,
        "stop_loss_pct": 2.0,
        "verify_minutes": 15,
        "max_coins": 60,
    },
    "추천5. 칼손절 초단타형": {
        "spike_tf": "5m",
        "min_vol_mult": 5.0,
        "min_ret_pct": 2.0,
        "target_pct": 3.0,
        "stop_loss_pct": 1.5,
        "verify_minutes": 10,
        "max_coins": 50,
    },
    "추천6. 15분봉 추세 확인형": {
        "spike_tf": "15m",
        "min_vol_mult": 3.5,
        "min_ret_pct": 2.0,
        "target_pct": 3.5,
        "stop_loss_pct": 2.0,
        "verify_minutes": 30,
        "max_coins": 80,
    },
}

_SETTING_KEYS = (
    "spike_tf",
    "min_vol_mult",
    "min_ret_pct",
    "target_pct",
    "stop_loss_pct",
    "verify_minutes",
    "max_coins",
)

_DEFAULT_PRESET_NAME = "추천1. 5분 초단타 셋팅"
_DEFAULT_PRESET_VALS = SCALP_PRESETS[_DEFAULT_PRESET_NAME]

if "scalp_preset_select" not in st.session_state:
    st.session_state["scalp_preset_select"] = _DEFAULT_PRESET_NAME
if "active_preset_name" not in st.session_state:
    st.session_state["active_preset_name"] = _DEFAULT_PRESET_NAME
for _k, _v in _DEFAULT_PRESET_VALS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _is_same_setting_val(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    return str(a) == str(b)


def _find_matching_preset_in_state() -> str | None:
    selected = st.session_state.get("scalp_preset_select")
    order = ([selected] if selected in SCALP_PRESETS else []) + [
        k for k in SCALP_PRESETS.keys() if k != selected
    ]
    for name in order:
        cfg = SCALP_PRESETS[name]
        if all(_is_same_setting_val(st.session_state.get(k), cfg[k]) for k in _SETTING_KEYS):
            return name
    return None


def _apply_selected_scalp_preset():
    chosen = st.session_state.get("scalp_preset_select")
    cfg = SCALP_PRESETS.get(chosen)
    if not cfg:
        return
    for k, v in cfg.items():
        st.session_state[k] = v
    st.session_state["active_preset_name"] = chosen


def _mark_custom_if_modified():
    matched = _find_matching_preset_in_state()
    if matched is not None:
        st.session_state["active_preset_name"] = matched
    else:
        st.session_state["active_preset_name"] = "커스텀"


# =====================================================================
# 화면 구성 (5–15분 초단타 중심 레이아웃)
# =====================================================================
st.title("⚡ 업비트 단타 & AI 스캐너")
st.caption(f"KST 기준 ({now_kst_str('%H:%M:%S')}) · 5–15분 초단타 수급 포착 + LightGBM AI 스캐너")

if "reset_nonce" not in st.session_state:
    st.session_state["reset_nonce"] = 0
if "last_reset_msg" not in st.session_state:
    st.session_state["last_reset_msg"] = None
if "scan_result_df" not in st.session_state:
    st.session_state["scan_result_df"] = None
if "watchlist_result_df" not in st.session_state:
    st.session_state["watchlist_result_df"] = None
if "scalp_result_df" not in st.session_state:
    st.session_state["scalp_result_df"] = None
if "stop_collect_requested" not in st.session_state:
    st.session_state["stop_collect_requested"] = False
if "collect_status_msg" not in st.session_state:
    st.session_state["collect_status_msg"] = None
if "train_status_msg" not in st.session_state:
    st.session_state["train_status_msg"] = None

with st.sidebar:
    st.header("⚙️ 제어 패널")
    top_n = st.slider("AI 표시 상위 코인 수", min_value=3, max_value=30, value=10, step=1)
    min_prob_3 = st.slider("AI 최소 +3% 확률 필터 (%)", min_value=30, max_value=85, value=55, step=5) / 100.0
    watch_min_prob_3 = st.slider("관심/대기 최소 +3% 확률 (%)", min_value=30, max_value=70, value=45, step=5) / 100.0
    success_target = st.selectbox("AI 성공 판정 목표", options=[3, 5, 10, 20], index=0)

    if st.button("🔄 화면 새로고침", width="stretch"):
        st.rerun()

    st.divider()
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        start_collect_clicked = st.button("📥 캔들 수집", width="stretch")
    with c_col2:
        if st.button("🛑 수집 중지", width="stretch"):
            st.session_state["stop_collect_requested"] = True
            st.session_state["collect_status_msg"] = "🛑 수집 중지 요청됨 — 현재 작업 마무리 후 멈춥니다."
            st.rerun()

    if st.session_state["collect_status_msg"]:
        st.caption(st.session_state["collect_status_msg"])

    if start_collect_clicked:
        st.session_state["stop_collect_requested"] = False
        st.session_state["collect_status_msg"] = None
        p_bar = st.progress(0.02)
        p_cap = st.empty()
        cb = _make_ui_progress_cb(p_bar, p_cap)
        try:
            def _is_stop_requested() -> bool:
                return bool(st.session_state.get("stop_collect_requested", False))

            run_collector_direct(progress_cb=cb, stop_check=_is_stop_requested, force=False)
            if _is_stop_requested():
                st.session_state["collect_status_msg"] = "🛑 수집이 중지되었습니다. (지금까지 수집된 데이터는 DB에 저장됨)"
            else:
                p_bar.progress(1.0)
                p_cap.caption("✅ 최신 캔들 수집 완료 (100%)")
                st.session_state["collect_status_msg"] = "✅ 1d / 240m / 60m 수집 완료"
            st.session_state["stop_collect_requested"] = False
            st.rerun()
        except Exception as exc:
            st.session_state["stop_collect_requested"] = False
            st.error(f"❌ 수집 오류: {type(exc).__name__}: {exc}")

    st.divider()
    if st.button("🧠 AI 모델 학습/재학습", width="stretch"):
        st.session_state["train_status_msg"] = None
        p_bar = st.progress(0.05)
        p_cap = st.empty()
        cb = _make_ui_progress_cb(p_bar, p_cap)
        try:
            run_training_direct(progress_cb=cb)
            p_bar.progress(1.0)
            p_cap.caption("✅ AI 모델 학습 완료 (100%)")
            st.session_state["train_status_msg"] = f"✅ AI 모델 학습 완료 ({now_kst_str('%H:%M:%S')})"
            st.rerun()
        except Exception as exc:
            st.session_state["train_status_msg"] = f"❌ 학습 실패 ({now_kst_str('%H:%M:%S')})"
            st.error(f"❌ 학습 오류: {type(exc).__name__}: {exc}")

    if st.session_state["train_status_msg"]:
        st.caption(st.session_state["train_status_msg"])

    st.divider()
    if st.button("🔍 지금 전 종목 AI 스캔", type="primary", width="stretch"):
        p_bar = st.progress(0.05)
        p_cap = st.empty()
        cb = _make_ui_progress_cb(p_bar, p_cap)
        try:
            scan_bundle = run_scan_direct(
                top_n=top_n,
                min_prob_3=min_prob_3,
                watchlist_n=8,
                watch_min_prob_3=watch_min_prob_3,
                progress_cb=cb,
            )
            st.session_state["scan_result_df"] = scan_bundle.get("recommended", pd.DataFrame())
            st.session_state["watchlist_result_df"] = scan_bundle.get("watchlist", pd.DataFrame())
            p_bar.progress(1.0)
            p_cap.caption("✅ 전 종목 스캔 완료 (100%)")
        except Exception as exc:
            st.error(f"❌ 스캔 오류: {type(exc).__name__}: {exc}")

    st.divider()
    if st.button("✅ AI 성공여부 검증", width="stretch"):
        p_bar = st.progress(0.10)
        p_cap = st.empty()
        cb = _make_ui_progress_cb(p_bar, p_cap)
        try:
            _call_with_optional_progress(verify_predictions, progress_cb=cb, success_target_pct=success_target)
            p_bar.progress(1.0)
            p_cap.caption("✅ AI 성공여부 검증 완료 (100%)")
            st.rerun()
        except Exception as exc:
            st.error(f"❌ AI 검증 오류: {type(exc).__name__}: {exc}")

    st.divider()
    st.subheader("🗑️ 로그 및 통계 초기화")
    nonce = st.session_state["reset_nonce"]
    reset_scalp_chk = st.checkbox("단타 통계 초기화 대상 포함", key=f"chk_reset_scalp_{nonce}")
    reset_ai_chk = st.checkbox("AI 통계 초기화 대상 포함", key=f"chk_reset_ai_{nonce}")
    confirm_reset_chk = st.checkbox("삭제 실행 확인", key=f"chk_confirm_reset_{nonce}")

    if st.session_state["last_reset_msg"]:
        st.success(st.session_state["last_reset_msg"])
        st.session_state["last_reset_msg"] = None

    can_reset = (reset_scalp_chk or reset_ai_chk) and confirm_reset_chk
    if st.button("🧹 선택 통계 초기화 실행", width="stretch", disabled=not can_reset):
        cleared = []
        if reset_scalp_chk:
            reset_scalp_logs()
            st.session_state["scalp_result_df"] = None
            cleared.append("단타 통계")
        if reset_ai_chk:
            reset_prediction_logs()
            st.session_state["scan_result_df"] = None
            st.session_state["watchlist_result_df"] = None
            cleared.append("AI 통계")

        st.session_state["reset_nonce"] = nonce + 1
        st.session_state["last_reset_msg"] = f"{' · '.join(cleared)} 초기화 완료!"
        st.rerun()

# =====================================================================
# 1. [상단 메인] 실시간 단타 포착 (추천 셋팅 선택 시 슬라이더 즉시 연동)
# =====================================================================
st.subheader("1. 실시간 단타 포착 (5–15분 내 +3–5% 초단타 세팅)")

st.selectbox(
    "추천 단타 세팅 선택 (선택 즉시 슬라이더 반영)",
    options=list(SCALP_PRESETS.keys()),
    key="scalp_preset_select",
    on_change=_apply_selected_scalp_preset,
)

r1_c1, r1_c2, r1_c3 = st.columns(3)
spike_tf = r1_c1.selectbox(
    "감지 봉",
    options=["5m", "15m"],
    key="spike_tf",
    on_change=_mark_custom_if_modified,
)
min_vol_mult = r1_c2.slider(
    "거래량 배율 (배)",
    min_value=1.5,
    max_value=10.0,
    step=0.5,
    key="min_vol_mult",
    on_change=_mark_custom_if_modified,
)
min_ret_pct = r1_c3.slider(
    "최소 상승률 (%)",
    min_value=0.5,
    max_value=10.0,
    step=0.5,
    key="min_ret_pct",
    on_change=_mark_custom_if_modified,
)

r2_c1, r2_c2, r2_c3 = st.columns(3)
target_pct = r2_c1.slider(
    "목표수익 (%)",
    min_value=1.0,
    max_value=10.0,
    step=0.5,
    key="target_pct",
    on_change=_mark_custom_if_modified,
)
stop_loss_pct = r2_c2.slider(
    "손절 기준 (%)",
    min_value=0.5,
    max_value=5.0,
    step=0.5,
    key="stop_loss_pct",
    on_change=_mark_custom_if_modified,
)
verify_minutes = r2_c3.slider(
    "검증 시간 (분)",
    min_value=10,
    max_value=60,
    step=5,
    key="verify_minutes",
    on_change=_mark_custom_if_modified,
)

max_coins = st.slider(
    "거래대금 상위 탐색 수",
    min_value=10,
    max_value=296,
    step=10,
    key="max_coins",
    on_change=_mark_custom_if_modified,
)

active_preset_label = st.session_state.get("active_preset_name") or _DEFAULT_PRESET_NAME
st.caption(
    f"현재 적용 기록 세팅: **{active_preset_label}** · "
    f"포착 시 저장되는 기준: **검증 {verify_minutes}분 / 손절 -{stop_loss_pct:.1f}%**"
)

b_col1, b_col2 = st.columns([2, 1])
with b_col1:
    if st.button("⚡ 지금 단타 종목 포착하기", type="primary", width="stretch"):
        s_bar = st.progress(0.05)
        s_cap = st.empty()
        s_cb = _make_ui_progress_cb(s_bar, s_cap)
        try:
            sp_df = detect_volume_spikes(
                timeframe=spike_tf,
                min_vol_mult=min_vol_mult,
                min_ret_pct=min_ret_pct,
                max_ret_pct=8.0,
                min_trade_krw=30_000_000,
                target_pct=target_pct,
                stop_loss_pct=stop_loss_pct,
                verify_minutes=verify_minutes,
                max_coins=max_coins,
                preset_name=active_preset_label,
                save_to_db=True,
                progress_cb=s_cb,
            )
            st.session_state["scalp_result_df"] = sp_df
            s_bar.progress(1.0)
            s_cap.caption(f"✅ 단타 포착 완료 [{active_preset_label} · {verify_minutes}분] (100%)")
        except Exception as exc:
            st.error(f"❌ 단타 포착 오류: {type(exc).__name__}: {exc}")

with b_col2:
    if st.button("✅ 단타 검증 (로그별 기준)", width="stretch"):
        v_bar = st.progress(0.05)
        v_cap = st.empty()
        v_cb = _make_ui_progress_cb(v_bar, v_cap)
        try:
            verify_scalp_predictions(
                expire_hours=verify_minutes / 60.0,
                stop_loss_pct=stop_loss_pct,
                progress_cb=v_cb,
            )
            v_bar.progress(1.0)
            v_cap.caption("✅ 단타 검증 완료 (100%)")
            st.rerun()
        except Exception as exc:
            st.error(f"❌ 단타 검증 오류: {type(exc).__name__}: {exc}")

spike_df = st.session_state.get("scalp_result_df")
if isinstance(spike_df, pd.DataFrame) and not spike_df.empty:
    formatted_spike = attach_korean_coin_names(spike_df, col_name="코인")
    formatted_spike = format_price_columns(
        formatted_spike,
        price_cols=["추천매수금액", "추천매도금액", "포착시점현재가"],
        int_cols=["거래대금(만원)", "검증기준(분)"],
    )
    formatted_spike = reorder_columns_front(
        formatted_spike,
        [
            "코인",
            "세팅(프리셋)",
            "검증기준(분)",
            "손절기준(%)",
            "추천매수금액",
            "추천매도금액",
            "거래량배율(배)",
            "상승률(%)",
            "추천매수타이밍",
            "포착시점현재가",
        ],
    )
    st.dataframe(formatted_spike, width="stretch", hide_index=True)
elif isinstance(spike_df, pd.DataFrame) and spike_df.empty:
    st.info("현재 조건에 맞는 단타 종목이 없습니다. 거래량 배율이나 최소 상승률을 낮춰 다시 눌러보세요.")

st.divider()

# =====================================================================
# 2. 단타 성공/실패 통계 및 포착 로그
# =====================================================================
st.subheader("2. 단타 포착 통계 & 로그")
scalp_stat = get_scalp_statistics()
if scalp_stat["total"] == 0:
    st.caption("아직 저장된 단타 포착 로그가 없습니다. **'지금 단타 종목 포착하기'**를 누르면 기록됩니다.")
else:
    s1, s2, s3 = st.columns(3)
    s1.metric("누적 단타 포착", f"{scalp_stat['total']:,}건")
    s2.metric("판정 완료", f"{scalp_stat['settled']:,}건")
    s3.metric("단타 실제 성공률", f"{scalp_stat['win_rate']}%")

    s4, s5, s6 = st.columns(3)
    s4.metric("성공", f"{scalp_stat['success']:,}건")
    s5.metric("실패", f"{scalp_stat['fail']:,}건")
    s6.metric("판정건 평균 최고수익", f"{scalp_stat['avg_max_gain']}%")

    if not scalp_stat["by_preset"].empty:
        st.caption("📊 추천 세팅(프리셋)별 단타 성적 비교")
        st.dataframe(scalp_stat["by_preset"], width="stretch", hide_index=True)

    f_c1, f_c2 = st.columns([2, 1])
    with f_c1:
        scalp_filter = st.radio(
            "단타 결과 필터",
            options=["전체", "성공", "실패", "대기중"],
            horizontal=True,
            key="scalp_filter_radio",
        )
    with f_c2:
        preset_options = ["전체"] + list(SCALP_PRESETS.keys()) + ["커스텀"]
        scalp_preset_filter = st.selectbox(
            "세팅(프리셋) 필터",
            options=preset_options,
            key="scalp_preset_filter_select",
        )

    scalp_logs_df = load_scalp_logs(
        limit=200,
        outcome_filter=scalp_filter,
        preset_filter=scalp_preset_filter,
    )
    st.dataframe(scalp_logs_df, width="stretch", hide_index=True)

    with st.expander("거래량 배율 구간별 단타 성공률 보기"):
        if not scalp_stat["by_vol"].empty:
            st.dataframe(scalp_stat["by_vol"], width="stretch", hide_index=True)

st.divider()

# =====================================================================
# 3. AI 스캔 결과 & 관심/대기 후보 & 누적 AI 추천 통계
# =====================================================================
st.subheader("3. AI 상승 확률 스캔 & 누적 추천 통계")
if not models_ready():
    st.info("아직 학습된 모델 파일(`models/lgb_target_*.txt`)이 없습니다. 왼쪽 메뉴에서 학습을 진행해 주세요.")

st.caption(f"🎯 **실전 추천 매수 기준:** +3% 확률 **{int(round(min_prob_3 * 100))}%** 이상 (로그 기록 대상)")
scan_df = st.session_state.get("scan_result_df")
if isinstance(scan_df, pd.DataFrame) and not scan_df.empty:
    formatted_scan = attach_korean_coin_names(scan_df, col_name="코인")
    formatted_scan = format_price_columns(
        formatted_scan,
        price_cols=["현재가", "추천매수가", "손절가", "목표+3%", "목표+5%", "목표+10%", "목표+20%"],
    )
    st.dataframe(formatted_scan, width="stretch", hide_index=True)
elif isinstance(scan_df, pd.DataFrame) and scan_df.empty:
    st.info(f"현재 +3% 확률 {int(round(min_prob_3 * 100))}% 기준을 충족하는 추천 매수 종목이 없습니다.")

watch_df = st.session_state.get("watchlist_result_df")
if isinstance(watch_df, pd.DataFrame):
    st.markdown(
        f"👀 **관심/대기 후보** (+3% 확률 **{int(round(watch_min_prob_3 * 100))}~{max(int(round(watch_min_prob_3 * 100)), int(round(min_prob_3 * 100)) - 1)}%** · 추세/수급/과열 필터 통과 · 통계 미포함)"
    )
    if not watch_df.empty:
        formatted_watch = attach_korean_coin_names(watch_df, col_name="코인")
        formatted_watch = format_price_columns(
            formatted_watch,
            price_cols=["현재가"],
        )
        st.dataframe(formatted_watch, width="stretch", hide_index=True)
    else:
        st.caption("현재 관심/대기 조건을 만족하는 후보 종목이 없습니다.")

with st.expander("누적 AI 추천 리스트 & 성공/실패 통계 보기"):
    stat_bundle = get_prediction_statistics()
    if stat_bundle["total"] == 0:
        st.caption("아직 저장된 AI 추천 코인 리스트가 없습니다.")
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("누적 추천", f"{stat_bundle['total']:,}건")
        k2.metric("판정 완료", f"{stat_bundle['settled']:,}건")
        k3.metric("성공", f"{stat_bundle['success']:,}건")
        k4.metric("실패", f"{stat_bundle['fail']:,}건")
        k5.metric("실제 성공률", f"{stat_bundle['win_rate']}%")

        reason_df = get_ai_reason_summary()
        if not reason_df.empty:
            st.dataframe(reason_df, width="stretch", hide_index=True)

        target_stats_df = stat_bundle["target_stats"]
        if not target_stats_df.empty:
            st.dataframe(target_stats_df, width="stretch", hide_index=True)

        filter_choice = st.radio(
            "AI 로그 필터",
            options=["전체", "성공", "실패", "대기중"],
            horizontal=True,
            key="ai_filter_radio",
        )
        logs_df = load_prediction_logs(limit=200, outcome_filter=filter_choice)
        st.dataframe(logs_df, width="stretch", hide_index=True)

with st.expander("DB 수집 현황 (1d / 240m / 60m) 보기"):
    stats_df = load_db_stats()
    if stats_df.empty:
        st.caption("아직 DB에 수집된 캔들 데이터가 없습니다.")
    else:
        cols = st.columns(len(stats_df))
        label_map = {"1d": "일봉 (1d)", "240m": "4시간봉 (240m)", "60m": "60분봉 (60m)"}
        for idx, (_, r) in enumerate(stats_df.iterrows()):
            with cols[idx]:
                tf_label = label_map.get(r["timeframe"], r["timeframe"])
                st.metric(
                    label=tf_label,
                    value=f"{int(r['candle_count']):,}개",
                    delta=f"{int(r['coin_count']):,}개 코인",
                )
                earliest_dt = _parse_dt_safe(r["earliest_kst"])
                latest_dt = _parse_dt_safe(r["latest_kst"])
                earliest_str = (
                    earliest_dt.strftime("%Y-%m-%d %H:%M")
                    if earliest_dt
                    else str(r["earliest_kst"])[:16].replace("T", " ")
                )
                latest_str = (
                    latest_dt.strftime("%Y-%m-%d %H:%M")
                    if latest_dt
                    else str(r["latest_kst"])[:16].replace("T", " ")
                )
                st.caption(f"기간: {earliest_str} ~ {latest_str}")