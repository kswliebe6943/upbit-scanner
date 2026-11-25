import os
import sqlite3
from contextlib import contextmanager
from typing import Any

import pandas as pd

DEFAULT_SQLITE_PATH = r"C:\upbit_Ai_scanner\crypto_data.db"


def get_database_url() -> str | None:
    url = os.environ.get("DATABASE_URL")
    if url and url.strip():
        return url.strip()
    try:
        import streamlit as st
        if "DATABASE_URL" in st.secrets:
            val = str(st.secrets["DATABASE_URL"]).strip()
            if val:
                return val
    except Exception:
        pass
    return None


def is_pg_enabled() -> bool:
    url = get_database_url()
    return bool(url and (url.startswith("postgres://") or url.startswith("postgresql://")))


def _adapt_sql(sql: str, backend: str) -> str:
    if backend == "pg":
        return sql.replace("?", "%s")
    return sql


@contextmanager
def open_log_db(sqlite_path: str = DEFAULT_SQLITE_PATH):
    if is_pg_enabled():
        import psycopg2
        conn = psycopg2.connect(get_database_url())
        try:
            yield conn, "pg"
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(sqlite_path, timeout=30.0, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            yield conn, "sqlite"
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def execute_log_sql(sql: str, params: tuple[Any, ...] | list[Any] = (), sqlite_path: str = DEFAULT_SQLITE_PATH):
    with open_log_db(sqlite_path) as (conn, backend):
        cur = conn.cursor()
        cur.execute(_adapt_sql(sql, backend), tuple(params))
        return cur


def executemany_log_sql(sql: str, seq_of_params: list[tuple[Any, ...]], sqlite_path: str = DEFAULT_SQLITE_PATH):
    if not seq_of_params:
        return
    with open_log_db(sqlite_path) as (conn, backend):
        cur = conn.cursor()
        cur.executemany(_adapt_sql(sql, backend), seq_of_params)


def read_log_df(sql: str, params: tuple[Any, ...] | list[Any] = (), sqlite_path: str = DEFAULT_SQLITE_PATH) -> pd.DataFrame:
    with open_log_db(sqlite_path) as (conn, backend):
        adapted = _adapt_sql(sql, backend)
        if backend == "pg":
            return pd.read_sql_query(adapted, conn, params=tuple(params) if params else None)
        return pd.read_sql_query(adapted, conn, params=tuple(params) if params else ())


def ensure_log_tables(sqlite_path: str = DEFAULT_SQLITE_PATH):
    with open_log_db(sqlite_path) as (conn, backend):
        cur = conn.cursor()
        if backend == "pg":
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS scalp_logs (
                    id SERIAL PRIMARY KEY,
                    detected_at TEXT NOT NULL,
                    preset_name TEXT NOT NULL DEFAULT '커스텀',
                    market TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    current_price DOUBLE PRECISION NOT NULL,
                    rec_buy_price DOUBLE PRECISION NOT NULL,
                    rec_sell_price DOUBLE PRECISION NOT NULL,
                    target_pct DOUBLE PRECISION NOT NULL,
                    stop_loss_pct DOUBLE PRECISION NOT NULL DEFAULT 2.0,
                    verify_minutes INTEGER NOT NULL DEFAULT 15,
                    timing TEXT NOT NULL,
                    ret_pct DOUBLE PRECISION NOT NULL,
                    vol_mult DOUBLE PRECISION NOT NULL,
                    trade_val_krw DOUBLE PRECISION NOT NULL,
                    bought INTEGER DEFAULT 0,
                    max_gain_pct DOUBLE PRECISION DEFAULT 0.0,
                    outcome TEXT NOT NULL DEFAULT 'PENDING',
                    status_detail TEXT NOT NULL DEFAULT '대기중',
                    verified_at TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id SERIAL PRIMARY KEY,
                    predicted_at TEXT NOT NULL,
                    market TEXT NOT NULL,
                    trade_style TEXT DEFAULT '단타',
                    current_price DOUBLE PRECISION NOT NULL,
                    rec_buy_price DOUBLE PRECISION NOT NULL,
                    stop_loss DOUBLE PRECISION NOT NULL,
                    target_3 DOUBLE PRECISION NOT NULL,
                    target_5 DOUBLE PRECISION NOT NULL,
                    target_10 DOUBLE PRECISION NOT NULL,
                    target_20 DOUBLE PRECISION NOT NULL,
                    prob_3 DOUBLE PRECISION DEFAULT 0.0,
                    prob_5 DOUBLE PRECISION DEFAULT 0.0,
                    prob_10 DOUBLE PRECISION DEFAULT 0.0,
                    prob_20 DOUBLE PRECISION DEFAULT 0.0,
                    expected_buy_window TEXT,
                    status TEXT DEFAULT 'PENDING',
                    outcome TEXT DEFAULT 'PENDING',
                    max_gain_pct DOUBLE PRECISION DEFAULT 0.0,
                    hit_3 INTEGER DEFAULT 0,
                    hit_5 INTEGER DEFAULT 0,
                    hit_10 INTEGER DEFAULT 0,
                    hit_20 INTEGER DEFAULT 0,
                    verified_at TEXT
                )
                """
            )
        else:
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    predicted_at TEXT NOT NULL,
                    market TEXT NOT NULL,
                    trade_style TEXT DEFAULT '단타',
                    current_price REAL NOT NULL,
                    rec_buy_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    target_3 REAL NOT NULL,
                    target_5 REAL NOT NULL,
                    target_10 REAL NOT NULL,
                    target_20 REAL NOT NULL,
                    prob_3 REAL DEFAULT 0.0,
                    prob_5 REAL DEFAULT 0.0,
                    prob_10 REAL DEFAULT 0.0,
                    prob_20 REAL DEFAULT 0.0,
                    expected_buy_window TEXT,
                    status TEXT DEFAULT 'PENDING',
                    outcome TEXT DEFAULT 'PENDING',
                    max_gain_pct REAL DEFAULT 0.0,
                    hit_3 INTEGER DEFAULT 0,
                    hit_5 INTEGER DEFAULT 0,
                    hit_10 INTEGER DEFAULT 0,
                    hit_20 INTEGER DEFAULT 0,
                    verified_at TEXT
                )
                """
            )