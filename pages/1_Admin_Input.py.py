import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

DB_FILE = "dpc_demo.sqlite3"


# -----------------------------
# DB helpers
# -----------------------------
def connect_db():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def ensure_db(conn):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT,
            session_name TEXT,
            lap INTEGER,
            total_laps INTEGER,
            flag_state TEXT,
            driver TEXT,
            team TEXT,
            country TEXT,
            grid_pos INTEGER,
            position INTEGER,
            odds_american INTEGER,
            fv_dollars REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            k TEXT PRIMARY KEY,
            v TEXT
        )
        """
    )
    conn.commit()


def set_state(conn, k, v):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO app_state (k, v) VALUES (?, ?) "
        "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (k, v),
    )
    conn.commit()


def get_state(conn, k, default=""):
    cur = conn.cursor()
    cur.execute("SELECT v FROM app_state WHERE k=?", (k,))
    row = cur.fetchone()
    return row[0] if row else default


def insert_ticks(conn, rows):
    if not rows:
        return
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO ticks (
            ts_utc, session_name, lap, total_laps, flag_state,
            driver, team, country, grid_pos, position,
            odds_american, fv_dollars
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


# -----------------------------
# Pricing model
# -----------------------------
@dataclass
class DemoParams:
    field_size: int
    sigma_min: float
    sigma_max: float
    caution_mult: float
    rating_strength: float


def american_to_prob(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def odds_to_rating(odds_series: pd.Series) -> pd.Series:
    p = odds_series.apply(american_to_prob).clip(1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    mu = np.nanmean(logit)
    sd = np.nanstd(logit)
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return ((logit - mu) / sd).fillna(0.0)


def sigma_from_progress(lap, total_laps, params, flag_state):
    progress = float(lap) / float(total_laps) if total_laps > 0 else 0
    sigma = params.sigma_max * (1 - progress) + params.sigma_min * progress
    if str(flag_state).lower().startswith("yellow"):
        sigma *= params.caution_mult
    return max(0.25, sigma)


def fair_value(position, lap, total_laps, rating, params, settlement, flag_state):
    sigma = sigma_from_progress(lap, total_laps, params, flag_state)
    mu = float(position) - float(params.rating_strength) * float(rating)
    xs = np.arange(1, params.field_size + 1, dtype=float)
    dist = np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    dist = dist / dist.sum()
    return float(np.dot(dist, settlement))


def build_settlement_grid(field_size, first_prize):
    idx = np.arange(0, field_size)
    raw = np.exp(-0.18 * idx)
    raw = raw / raw[0]
    return first_prize * raw


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Admin Input — DPC Demo", layout="wide")
st.title("Admin Input — DPC Demo")

conn = connect_db()
ensure_db(conn)

# -----------------------------
# Session
# -----------------------------
st.subheader("Session Setup")
session_name = st.text_input(
    "Active Session",
    value=get_state(conn, "ACTIVE_SESSION", "New Session"),
)

if st.button("Set Active Session"):
    set_state(conn, "ACTIVE_SESSION", session_name)
    st.success("Active session updated.")

# -----------------------------
# Race state
# -----------------------------
st.subheader("Race State")
c1, c2, c3 = st.columns(3)
with c1:
    total_laps = st.number_input("Total Laps", 1, 500, 100)
with c2:
    lap = st.number_input("Current Lap", 0, total_laps, 0)
with c3:
    flag_state = st.selectbox("Flag State", ["Green", "Yellow (caution)"])

field_size = st.number_input("Field Size", 10, 40, 27)
first_prize = st.number_input("1st Place Settlement ($)", 100.0, 10000.0, 1000.0)

settlement = build_settlement_grid(field_size, first_prize)

# -----------------------------
# Drivers
# -----------------------------
st.subheader("Drivers")

if "drivers_df" not in st.session_state:
    st.session_state["drivers_df"] = pd.DataFrame(
        columns=["driver", "team", "country", "grid_pos", "position", "odds_american"]
    )

df = st.data_editor(
    st.session_state["drivers_df"],
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="drivers_editor",
)
st.session_state["drivers_df"] = df.copy()

# -----------------------------
# Load roster from DB
# -----------------------------
st.divider()
if st.button("Load drivers from last published marks"):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ts_utc, driver, team, country, grid_pos, position, odds_american
        FROM ticks
        WHERE session_name = ?
        ORDER BY ts_utc DESC
        """,
        (session_name,),
    )
    rows = cur.fetchall()

    if rows:
        df_load = pd.DataFrame(
            rows,
            columns=["ts_utc", "driver", "team", "country", "grid_pos", "position", "odds_american"],
        )
        df_load = df_load.drop_duplicates(subset=["driver"], keep="first")
        df_load = df_load[["driver", "team", "country", "grid_pos", "position", "odds_american"]]
        df_load = df_load.sort_values("grid_pos").reset_index(drop=True)
        st.session_state["drivers_df"] = df_load
        st.success(f"Loaded {len(df_load)} drivers.")
        st.rerun()
    else:
        st.warning("No published marks found for this session.")

# -----------------------------
# Publishing
# -----------------------------
st.subheader("Publishing")

def publish_snapshot(prerace=False):
    df_pub = st.session_state["drivers_df"].copy()
    df_pub = df_pub.dropna(subset=["driver", "grid_pos", "position"])
    if df_pub.empty:
        st.warning("No valid rows to publish.")
        return

    params = DemoParams(field_size, 2.0, 8.0, 1.4, 1.0)
    df_pub["rating"] = odds_to_rating(df_pub["odds_american"])
    df_pub["dpc_fv"] = df_pub.apply(
        lambda r: fair_value(
            r["position"],
            lap,
            total_laps,
            r["rating"],
            params,
            settlement,
            flag_state,
        ),
        axis=1,
    )

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for _, r in df_pub.iterrows():
        rows.append(
            (
                ts,
                session_name,
                0 if prerace else lap,
                total_laps,
                "Green" if prerace else flag_state,
                r["driver"],
                r["team"],
                r["country"],
                int(r["grid_pos"]),
                int(r["grid_pos"]) if prerace else int(r["position"]),
                int(r["odds_american"]) if pd.notna(r["odds_american"]) else None,
                r["dpc_fv"],
            )
        )
    insert_ticks(conn, rows)
    st.success(f"Published {len(rows)} marks.")

col1, col2 = st.columns(2)
with col1:
    if st.button("Initialize Pre-Race Mark"):
        publish_snapshot(prerace=True)

with col2:
    if st.button("Publish Snapshot"):
        publish_snapshot(prerace=False)