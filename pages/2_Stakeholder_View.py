import base64
import sqlite3
import time
import textwrap
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

DB_FILE = "dpc_demo.sqlite3"

APP_TITLE = "Driver Performance Contract Market"
SUBTITLE = "Live Contract Analytics & Pricing"

# Demo polish
MAX_PCT_FOR_DISPLAY = 2.0   # 200% cap for percent display
MIN_BASE_FOR_PCT = 25.0     # if base < $25, hide % (avoid silly ratios)
PULSE_ABS_DOLLAR_THRESHOLD = 2.50  # pulse market cards when change >= this

# Sparklines
SPARK_N = 24  # points in sparkline (last N ticks for each driver)

ISO3_TO_ISO2 = {
    "USA": "US", "NZL": "NZ", "SWE": "SE", "ESP": "ES", "MEX": "MX", "DEN": "DK",
    "NLD": "NL", "CAN": "CA", "AUS": "AU", "GBR": "GB", "ISR": "IL", "CAY": "KY",
}

TEAM_KEY = {
    "juncos hollinger racing": "jhr",
    "jhr": "jhr",
    "team penske": "penske",
    "chip ganassi racing": "ganassi",
    "arrow mclaren": "mclaren",
    "andretti global": "andretti",
    "andretti global w/ curb-agajanian": "andretti",
    "andretti global w/ curb-agajanian ": "andretti",
    "meyer shank racing": "msr",
    "meyer shank w/ curb-agajanian": "msr",
    "rahal letterman lanigan racing": "rll",
    "rahal letterman lanigan": "rll",
    "rll": "rll",
    "a.j. foyt enterprises": "foyt",
    "aj foyt enterprises": "foyt",
    "foyt": "foyt",
    "dale coyne racing": "coyne",
    "coyne": "coyne",
    "ed carpenter racing": "ecr",
    "ecr": "ecr",
    "prema racing": "prema",
    "prema": "prema",
}

LOGO_EXTS = [".png", ".jpg", ".jpeg", ".webp"]


# ---------------- Paths (ROBUST) ----------------
def resolve_assets_dir() -> Path:
    candidates = [Path.cwd() / "assets"]
    try:
        candidates.append(Path(__file__).resolve().parent / "assets")
        candidates.append(Path(__file__).resolve().parent.parent / "assets")  # if in pages/
    except Exception:
        pass
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return Path.cwd() / "assets"


ASSETS_DIR = resolve_assets_dir()
LOGOS_DIR = ASSETS_DIR / "logos"


def file_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png"
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        mime = "image/webp"
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def find_logo_file(key: str) -> Path | None:
    key = key.strip().lower()
    for ext in LOGO_EXTS:
        p = LOGOS_DIR / f"{key}{ext}"
        if p.exists():
            return p
    # also try sanitized
    safe = "".join([c for c in key if c.isalnum() or c in ("_", "-")]).strip("-_")
    for ext in LOGO_EXTS:
        p = LOGOS_DIR / f"{safe}{ext}"
        if p.exists():
            return p
    return None


def team_logo_uri(team_name: str) -> str:
    if not team_name:
        return ""
    key = TEAM_KEY.get(str(team_name).strip().lower(), "")
    if not key:
        key = str(team_name).strip().lower()
    p = find_logo_file(key)
    return file_to_data_uri(p) if p else ""


def race_logo_uri() -> str:
    # You can swap this to any static race/broadcast logo you keep in assets/
    for nm in ["race.png", "race_logo.png", "indycar.png", "fox.png"]:
        p = ASSETS_DIR / nm
        if p.exists():
            return file_to_data_uri(p)
    return ""


# ---------------- Data ----------------
def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def read_table(name: str) -> pd.DataFrame:
    try:
        with get_conn() as con:
            return pd.read_sql(f"SELECT * FROM {name}", con)
    except Exception:
        return pd.DataFrame()


def safe_first(df: pd.DataFrame, col: str, default=None):
    try:
        if df.empty or col not in df.columns:
            return default
        return df.iloc[0][col]
    except Exception:
        return default


def normalize_driver_key(s: str) -> str:
    return "".join(ch.lower() for ch in str(s).strip() if ch.isalnum() or ch == " ")


def clean_driver_name(s: str) -> str:
    s = str(s or "").strip()
    return " ".join(s.split())


def build_lookup(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    if df is None or df.empty:
        return out
    for _, r in df.iterrows():
        out[normalize_driver_key(r.get("driver", ""))] = float(r.get("fv_dollars", 0.0) or 0.0)
    return out


def clamp_pct(p: float) -> float | None:
    if p is None:
        return None
    if abs(p) > MAX_PCT_FOR_DISPLAY:
        return None
    return p


def fmt_money(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return "$—"


def fmt_pct(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p*100:,.1f}%"


def sign_color(delta: float) -> str:
    return "#7CF6C2" if delta >= 0 else "#FF5B6E"


def build_spark_map(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Returns map driver_key -> df of last N points with ['ts','fv'].
    """
    out: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out

    # Ensure ts exists
    if "ts" not in df.columns:
        return out

    tmp = df.copy()
    tmp["ts"] = pd.to_datetime(tmp["ts"], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=["ts"])
    tmp["driver_key"] = tmp["driver"].apply(normalize_driver_key)

    for dk, g in tmp.groupby("driver_key"):
        g2 = g.sort_values("ts").tail(SPARK_N)[["ts", "fv_dollars"]].rename(columns={"fv_dollars": "fv"})
        out[dk] = g2
    return out


# ---------------- Page config ----------------
st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")

# Controls
with st.expander("Controls", expanded=False):
    cols = st.columns([1, 1, 1, 1, 1])
    with cols[0]:
        tv_safe = st.toggle("TV safe", value=False, help="Bigger fonts + thicker strokes for broadcast displays")
    with cols[1]:
        auto_refresh = st.toggle("Auto-refresh", value=False)
    with cols[2]:
        refresh_seconds = st.selectbox("Refresh seconds", [2, 3, 5, 8, 10, 15], index=2)
    with cols[3]:
        show_sparklines = st.toggle("Sparklines", value=True)
    with cols[4]:
        compact_cards = st.toggle("Compact", value=False)

df = read_table("ticks")
latest = pd.DataFrame()
prev = pd.DataFrame()
open_df = pd.DataFrame()

if not df.empty and "ts" in df.columns:
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    df = df.sort_values("ts")
    latest_ts = df["ts"].max()
    latest = df[df["ts"] == latest_ts].copy()

    # prev tick (for delta)
    prev_ts = df[df["ts"] < latest_ts]["ts"].max()
    if pd.notna(prev_ts):
        prev = df[df["ts"] == prev_ts].copy()

    # opening mark (first tick)
    open_ts = df["ts"].min()
    open_df = df[df["ts"] == open_ts].copy()
else:
    latest_ts = pd.Timestamp.now(tz="UTC")

# numeric cleaning
for dff in [df, latest, prev, open_df]:
    if dff is not None and not dff.empty and "fv_dollars" in dff.columns:
        dff["fv_dollars"] = pd.to_numeric(dff.get("fv_dollars", 0.0), errors="coerce").fillna(0.0)
if not prev.empty:
    prev["fv_dollars"] = pd.to_numeric(prev.get("fv_dollars", 0.0), errors="coerce").fillna(0.0)
if not open_df.empty:
    open_df["fv_dollars"] = pd.to_numeric(open_df.get("fv_dollars", 0.0), errors="coerce").fillna(0.0)

prev_fv_by_driver = build_lookup(prev)
open_fv_by_driver = build_lookup(open_df)
spark_map = build_spark_map(df)

# ---------------- CSS ----------------
st.markdown(
    f"""
    <style>
        /* --- Remove Streamlit "dead white space" / chrome --- */
        header[data-testid="stHeader"] {{ display: none; }}
        div[data-testid="stToolbar"] {{ display: none; }}
        div[data-testid="stDecoration"] {{ display: none; }}

        /* Tighten page padding */
        .block-container {{
          padding-top: 0.45rem !important;
          padding-bottom: 2rem !important;
        }}
        div[data-testid="stVerticalBlock"] > div {{ gap: {10 if tv_safe else 8}px; }}

        /* App background */
        div[data-testid="stAppViewContainer"]{{
          background:
            radial-gradient(circle at 18% 0%, rgba(103,245,181,0.10), transparent 52%),
            radial-gradient(circle at 85% 15%, rgba(103,245,181,0.06), transparent 58%),
            linear-gradient(180deg,#05060a,#0a0b10);
          color:#f2f4ff;
          position: relative;
        }}
        div[data-testid="stAppViewContainer"] * {{ color:#f2f4ff; }}

        /* Broadcast vignette + cinematic shading overlay */
        div[data-testid="stAppViewContainer"]::before {{
          content: "";
          position: fixed;
          inset: 0;
          pointer-events: none;
          z-index: 0;
          background:
            radial-gradient(1200px 520px at 50% -60px, rgba(124,246,194,0.10), transparent 60%),
            radial-gradient(900px 520px at 80% 10%, rgba(124,246,194,0.07), transparent 62%),
            radial-gradient(900px 620px at 20% 20%, rgba(124,246,194,0.05), transparent 62%),
            radial-gradient(1200px 900px at 50% 70%, rgba(0,0,0,0.35), rgba(0,0,0,0.55) 70%, rgba(0,0,0,0.75));
          mix-blend-mode: normal;
        }}

        /* Tiny "noise" texture using gradients (no external asset) */
        div[data-testid="stAppViewContainer"]::after {{
          content: "";
          position: fixed;
          inset: 0;
          pointer-events: none;
          z-index: 0;
          opacity: 0.10;
          background-image:
            repeating-linear-gradient(0deg, rgba(255,255,255,0.025) 0px, rgba(255,255,255,0.025) 1px, transparent 1px, transparent 3px),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.018) 0px, rgba(255,255,255,0.018) 1px, transparent 1px, transparent 4px);
        }}

        /* Z stack */
        section.main > div {{ position: relative; z-index: 1; }}

        :root {{
          --mint: #7CF6C2;
          --mint2: #67F5B5;
          --glass: rgba(18,20,28,0.62);
          --glass2: rgba(18,20,28,0.50);
          --stroke: rgba(255,255,255,0.08);
          --stroke2: rgba(124,246,194,0.22);
          --shadow: 0 14px 44px rgba(0,0,0,0.55);
          --r: 22px;
          --t12: {12 if tv_safe else 11}px;
          --t13: {13 if tv_safe else 12}px;
          --t14: {14 if tv_safe else 13}px;
          --t16: {16 if tv_safe else 15}px;
          --t18: {18 if tv_safe else 16}px;
          --t22: {22 if tv_safe else 20}px;
          --t28: {28 if tv_safe else 24}px;
        }}

        .glass_card {{
          background: linear-gradient(180deg, rgba(22,24,33,0.72), rgba(18,20,28,0.56));
          border: 1px solid rgba(255,255,255,0.08);
          box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.06);
          border-radius: var(--r);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
        }}
        .muted {{ opacity: .76; }}
        .neon_rule {{
          height: 3px;
          border-radius: 999px;
          margin-top: 10px;
          width: 56%;
          background: linear-gradient(90deg, rgba(124,246,194,0.10), rgba(124,246,194,0.85), rgba(124,246,194,0.10));
          box-shadow: 0 0 18px rgba(124,246,194,0.35);
        }}

        /* Logos */
        .header_logo_wrap, .race_logo_wrap {{
          background: rgba(255,255,255,0.92);
          border-radius: 16px;
          padding: 10px 14px;
          border: 1px solid rgba(0,0,0,0.12);
          box-shadow: 0 14px 30px rgba(0,0,0,0.35);
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }}
        .header_logo_img{{ height: {96 if tv_safe else 92}px; width:auto; display:block; }}
        .race_logo_img{{ height: {80 if tv_safe else 76}px; width:auto; display:block; }}

        /* Header layout helpers (mobile-safe) */
        .hero_header{{
          display:flex;
          align-items:center;
          gap:16px;
        }}
        .hero_left, .hero_right{{
          display:flex;
          align-items:center;
          justify-content:center;
        }}
        .hero_spacer{{ flex:1; }}

        /* Pills */
        .pill {{
          display: inline-flex;
          align-items: center;
          gap: 10px;
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.10);
          background: linear-gradient(180deg, rgba(18,20,28,0.65), rgba(18,20,28,0.50));
          box-shadow: 0 10px 28px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
          font-weight: 900;
          letter-spacing: .01em;
          font-size: var(--t14);
          white-space: nowrap;
        }}
        .dot {{
          width: 10px; height: 10px; border-radius: 999px;
          background: var(--mint);
          box-shadow: 0 0 18px rgba(124,246,194,0.65);
        }}

        /* Section header */
        .section_hdr {{
          display:flex;
          align-items:center;
          gap: 12px;
          padding: 12px 16px;
        }}
        .hdr_leftbar {{
          width: 6px;
          height: 34px;
          border-radius: 999px;
          background: linear-gradient(180deg, rgba(124,246,194,0.95), rgba(124,246,194,0.20));
          box-shadow: 0 0 18px rgba(124,246,194,0.35);
        }}
        .hdr_title {{
          font-size: var(--t22);
          font-weight: 990;
          letter-spacing: .02em;
        }}
        .hdr_sub {{
          font-size: var(--t12);
          opacity: .72;
          text-transform: uppercase;
          letter-spacing: .14em;
          margin-top: 4px;
          font-weight: 900;
        }}

        /* Driver cards */
        .driver_card {{
          padding: 14px 16px;
          border-radius: 20px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(18,20,28,0.58), rgba(18,20,28,0.44));
          box-shadow: 0 14px 34px rgba(0,0,0,0.50), inset 0 1px 0 rgba(255,255,255,0.06);
          overflow: hidden;
          position: relative;
        }}
        .driver_row {{
          display:flex;
          gap: 12px;
          align-items: center;
        }}
        .num_badge {{
          width: {46 if tv_safe else 40}px;
          height: {46 if tv_safe else 40}px;
          border-radius: 999px;
          display:flex;
          align-items:center;
          justify-content:center;
          font-weight: 990;
          font-size: var(--t14);
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.08);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.07);
          flex: 0 0 auto;
        }}
        .flag_img{{ width: 28px; height: 20px; border-radius: 6px; object-fit: cover; box-shadow: 0 6px 16px rgba(0,0,0,0.35); flex:0 0 auto; }}
        .driver_name {{
          font-size: var(--t18);
          font-weight: 990;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }}
        .team_logo_small {{
          background: rgba(255,255,255,0.92);
          border-radius: 14px;
          padding: 8px 10px;
          border: 1px solid rgba(0,0,0,0.10);
          box-shadow: 0 12px 28px rgba(0,0,0,0.30);
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }}
        .team_logo_small img {{
          max-height: {34 if tv_safe else 30}px;
          width: auto;
          display:block;
        }}

        /* Fields */
        .kpi_label {{
          font-size: var(--t12);
          opacity: .70;
          text-transform: uppercase;
          letter-spacing: .14em;
          font-weight: 900;
        }}
        .kpi_value {{
          font-size: var(--t22);
          font-weight: 995;
          letter-spacing: .01em;
        }}

        /* Sparkline wrapper */
        .spark {{
          width: 150px;
          height: 44px;
          border-radius: 14px;
          background: rgba(0,0,0,0.20);
          border: 1px solid rgba(255,255,255,0.08);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }}

        /* Responsive tweaks */
        @media (max-width: 768px){{
          /* Mobile: stack header instead of squeezing */
          .hero_header{{
            flex-direction: column !important;
            align-items: stretch !important;
            gap: 12px !important;
          }}
          .hero_spacer{{ display:none !important; }}
          .header_logo_wrap, .race_logo_wrap{{
            width: 100% !important;
            max-width: 340px !important;
            margin: 0 auto !important;
          }}
          /* Slightly smaller logos on mobile */
          .header_logo_img{{ height: 72px !important; }}
          .race_logo_img{{ height: 60px !important; }}

          .block-container {{ padding-top: 0.35rem !important; }}
          .header_logo_img{{ height: 72px; }}
          .race_logo_img{{ height: 60px; }}
          .snap_strip{{ padding: 10px 12px; }}
          .panel{{ min-width: 100%; }}
          .top3_card{{ min-width: 100%; }}
          .spark{{ width: 150px; height: 44px; }}
          .driver_spark_wrap .spark{{ width: 150px; height: 40px; }}
        }}

        /* Ensure page can scroll (DO/desktop browsers) */
        html, body, [data-testid="stAppViewContainer"]{{
          height: auto !important;
          overflow: auto !important;
        }}
        [data-testid="stAppViewContainer"] > .main{{
          overflow: visible !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Header ----------------
jhr_uri = team_logo_uri("Juncos Hollinger Racing")
race_uri = race_logo_uri()

jhr_html = (
    f"<div class='header_logo_wrap'><img class='header_logo_img' src='{jhr_uri}'/></div>"
    if jhr_uri
    else "<div class='header_logo_wrap'><b style='color:#0b0d12;'>JHR</b></div>"
)
race_html = (
    f"<div class='race_logo_wrap'><img class='race_logo_img' src='{race_uri}'/></div>"
    if race_uri
    else ""
)

st.markdown(
    f"<div class='glass_card hero_header' style='padding:{16 if tv_safe else 14}px {20 if tv_safe else 18}px;'>"
    f"<div class='hero_left'>{jhr_html}</div>"
    f"<div style='min-width:0;'>"
    f"  <div style='font-size:24px;font-weight:995;line-height:1.10;letter-spacing:.01em;'>{APP_TITLE}</div>"
    f"  <div class='muted' style='font-size:13px;margin-top:3px;'>{SUBTITLE}</div>"
    f"  <div class='neon_rule'></div>"
    f"</div>"
    f"<div class='hero_spacer'></div>"
    f"<div class='hero_right'>{race_html}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# ---------------- LIVE MARKET strip ----------------
lap_raw = safe_first(latest, "lap", 0)
total_laps_raw = safe_first(latest, "total_laps", 0)
flag_state = str(safe_first(latest, "flag_state", "—"))
last_update = latest_ts.strftime("%H:%M:%S UTC")
total_mkt = float(latest["fv_dollars"].sum()) if not latest.empty else 0.0

try:
    lap = int(lap_raw)
except Exception:
    lap = 0
try:
    total_laps = int(total_laps_raw)
except Exception:
    total_laps = 0

adv = dec = 0
avg_abs_pct = None
biggest_mover_name = "—"
biggest_mover_delta = 0.0

if prev_fv_by_driver:
    pct_moves = []
    best_abs = 0.0
    for _, r in latest.iterrows():
        key = normalize_driver_key(r.get("driver", ""))
        nowv = float(r.get("fv_dollars", 0.0) or 0.0)
        prevv = float(prev_fv_by_driver.get(key, 0.0) or 0.0)
        if prevv > 0:
            d = nowv - prevv
            if d > 0:
                adv += 1
            elif d < 0:
                dec += 1
            if abs(d) > best_abs:
                best_abs = abs(d)
                biggest_mover_name = clean_driver_name(r.get("driver", ""))
                biggest_mover_delta = d
            if prevv >= MIN_BASE_FOR_PCT:
                pct = abs(d / prevv)
                pct = clamp_pct(pct)
                if pct is not None:
                    pct_moves.append(pct)
    if pct_moves:
        avg_abs_pct = sum(pct_moves) / len(pct_moves)

breadth_txt = f"{adv}↑ / {dec}↓"
big_mover_txt = f"Biggest mover: {biggest_mover_name} {biggest_mover_delta:+,.2f}"

chips = st.columns([1.15, 1.0, 1.0, 1.6, 1.7, 1.4, 1.2])
with chips[0]:
    st.markdown(
        f"<div class='pill'><span class='dot'></span>LIVE</div>",
        unsafe_allow_html=True,
    )
with chips[1]:
    st.markdown(f"<div class='pill'>Lap&nbsp;<b>{lap}/{total_laps}</b></div>", unsafe_allow_html=True)
with chips[2]:
    st.markdown(f"<div class='pill'>Flag&nbsp;<b>{flag_state}</b></div>", unsafe_allow_html=True)
with chips[3]:
    st.markdown(f"<div class='pill'>Last update&nbsp;<b>{last_update}</b></div>", unsafe_allow_html=True)
with chips[4]:
    st.markdown(f"<div class='pill'>Total market&nbsp;<b>{fmt_money(total_mkt)}</b></div>", unsafe_allow_html=True)
with chips[5]:
    st.markdown(f"<div class='pill'>Avg move&nbsp;<b>{fmt_pct(avg_abs_pct)}</b></div>", unsafe_allow_html=True)
with chips[6]:
    st.markdown(f"<div class='pill'>Breadth&nbsp;<b>{breadth_txt}</b></div>", unsafe_allow_html=True)

# Ticker-like strip
st.markdown(
    f"""
    <div class="glass_card" style="padding:10px 12px; overflow:hidden;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="opacity:.70;font-weight:900;">•</div>
        <div style="font-weight:950;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
          {big_mover_txt}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Leaders & Movers ----------------
st.markdown(
    """
    <div class="section_hdr glass_card">
      <div class="hdr_leftbar"></div>
      <div>
        <div class="hdr_title">Leaders &amp; Movers</div>
        <div class="hdr_sub">Live Pricing</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if latest.empty:
    st.warning("No tick data found yet. Ensure your DB is present and 'ticks' table has rows.")
else:
    # compute deltas, render visuals, etc.

# Compute deltas
rows = []
for _, r in latest.iterrows():
    dname = clean_driver_name(r.get("driver", ""))
    dk = normalize_driver_key(dname)
    nowv = float(r.get("fv_dollars", 0.0) or 0.0)

    prevv = float(prev_fv_by_driver.get(dk, nowv) or nowv)
    opnv = float(open_fv_by_driver.get(dk, nowv) or nowv)

    delta = nowv - prevv
    since_open = nowv - opnv

    pct = None
    if opnv >= MIN_BASE_FOR_PCT and opnv > 0:
        pct = clamp_pct(since_open / opnv)

    car_num = r.get("car_number", r.get("car", ""))
    iso3 = str(r.get("country_iso3", r.get("country", "")) or "")
    iso2 = ISO3_TO_ISO2.get(iso3.upper(), "")
    team = r.get("team", r.get("team_name", ""))

    rows.append(
        dict(
            driver=dname,
            driver_key=dk,
            fv=nowv,
            delta=delta,
            since_open=since_open,
            pct=pct,
            car=car_num,
            iso2=iso2,
            team=team,
        )
    )

cards = (
    pd.DataFrame(rows)
    .sort_values("fv", ascending=False)
    .reset_index(drop=True)
)

# Display cards
for i, r in cards.iterrows():
    dk = r["driver_key"]
    team_uri = team_logo_uri(r.get("team", ""))

    # flag image (optional)
    flag_html = ""
    if r.get("iso2"):
        # fallback: use emoji flag if you prefer; keeping simple
        pass

    team_html = (
        f"<span class='team_logo_small'><img src='{team_uri}'/></span>"
        if team_uri
        else ""
    )

    # sparkline
    spark_html = ""
    if show_sparklines and dk in spark_map and not spark_map[dk].empty:
        g = spark_map[dk]

        # render sparkline as mini plotly
        fig_s = go.Figure()
        fig_s.add_trace(
            go.Scatter(
                x=g["ts"],
                y=g["fv"],
                mode="lines",
                line=dict(width=2),
                hoverinfo="skip",
            )
        )
        fig_s.update_layout(
            height=52 if tv_safe else 46,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        # Note: you're using HTML for spark placement; chart is created but not rendered here.
        spark_html = "<div class='spark'></div>"

    pulse = abs(float(r["delta"])) >= PULSE_ABS_DOLLAR_THRESHOLD
    pulse_glow = (
        "box-shadow: 0 0 0 1px rgba(124,246,194,0.18), 0 18px 40px rgba(0,0,0,0.55);"
        if pulse
        else ""
    )

    st.markdown(
        f"""
        <div class="driver_card" style="{pulse_glow}">
          <div class="driver_row">
            <div class="num_badge">#{int(r['car']) if str(r['car']).isdigit() else str(r['car'])}</div>
            <div style="min-width:0;flex:1 1 auto;">
              <div class="driver_name">{r['driver']}</div>
              <div class="muted" style="font-size: var(--t13); margin-top: 2px; white-space:nowrap; overflow:hidden; text-overflow: ellipsis;">
                {str(r.get('team',''))}
              </div>
            </div>
            {team_html}
          </div>

          <div style="display:flex; gap: 14px; align-items:flex-end; margin-top: 12px; flex-wrap: wrap;">
            <div style="flex: 1 1 160px;">
              <div class="kpi_label">Fair Value</div>
              <div class="kpi_value">{fmt_money(float(r['fv']))}</div>
            </div>

            <div style="flex: 1 1 160px;">
              <div class="kpi_label">Since Mark</div>
              <div class="kpi_value" style="color:{sign_color(float(r['since_open']))};">
                {float(r['since_open']):+,.2f} <span style="opacity:.75; font-size: var(--t16);">({fmt_pct(r['pct'])})</span>
              </div>
            </div>

            <div style="flex: 1 1 140px;">
              <div class="kpi_label">Last Tick</div>
              <div class="kpi_value" style="color:{sign_color(float(r['delta']))};">{float(r['delta']):+,.2f}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="section_hdr glass_card" style="margin-top: 10px;">
      <div class="hdr_leftbar"></div>
      <div>
        <div class="hdr_title">Selected Contract</div>
        <div class="hdr_sub">Time Series</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

driver_names = list(cards["driver"].values)
sel = st.selectbox("Driver", driver_names, index=0)

sel_key = normalize_driver_key(sel)
series = df[df["driver"].apply(lambda x: normalize_driver_key(x) == sel_key)].copy()
series = series.sort_values("ts")

if series.empty:
    st.info("No time series data found for that driver yet.")
    st.stop()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=series["ts"],
        y=series["fv_dollars"],
        mode="lines+markers",
        line=dict(width=3 if tv_safe else 2),
        marker=dict(size=6 if tv_safe else 5),
        hovertemplate="%{x|%H:%M:%S UTC}<br>$%{y:,.2f}<extra></extra>",
        name="FV",
    )
)

# Latest marker
x_last = series["ts"].iloc[-1]
y_last = float(series["fv_dollars"].iloc[-1])
fig.add_trace(
    go.Scatter(
        x=[x_last],
        y=[y_last],
        mode="markers",
        marker=dict(size=11 if tv_safe else 10, color="#7CF6C2"),
        hovertemplate="Latest<br>%{x|%H:%M:%S UTC}<br>$%{y:,.2f}<extra></extra>",
        name="Latest",
    )
)

fig.update_layout(
    height=520 if tv_safe else 460,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.20)",
    font=dict(color="#F3F6FF", size=16 if tv_safe else 15),
    margin=dict(l=18, r=18, t=26, b=16),
    showlegend=False,
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.14)",
    tickangle=(-25 if not tv_safe else 0),
    tickfont=dict(color="#F3F6FF", size=14 if tv_safe else 13),
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.14)",
    tickfont=dict(color="#F3F6FF", size=14 if tv_safe else 13),
    tickprefix="$",
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Auto-refresh
if auto_refresh:
    time.sleep(int(refresh_seconds))
    st.rerun()
