import base64
import sqlite3
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

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
RACE_LOGO_FILE = ASSETS_DIR / "race.png"


# ---------------- DB ----------------
def connect_db():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def read_ticks(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM ticks ORDER BY id ASC", conn)


# ---------------- Helpers ----------------
def clean_driver_name(driver: str) -> str:
    s = str(driver or "").strip()
    parts = s.split()
    if len(parts) >= 2:
        p0 = parts[0]
        if len(p0) in (2, 3) and p0.isalpha():
            return " ".join(parts[1:]).strip()
    return s


def normalize_driver_key(driver: str) -> str:
    return str(driver or "").strip()


def flag_url(country: str) -> str:
    if not country:
        return ""
    c = str(country).strip().upper()
    if len(c) == 3 and c in ISO3_TO_ISO2:
        c = ISO3_TO_ISO2[c]
    if len(c) != 2:
        return ""
    return f"https://flagcdn.com/w40/{c.lower()}.png"


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


@st.cache_data(show_spinner=False, ttl=3600)
def _file_to_data_uri(path_str: str) -> str:
    path = Path(path_str)
    data = _read_bytes(path)
    ext = path.suffix.lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def find_logo_file(base_key: str) -> Path | None:
    if not base_key:
        return None
    for ext in LOGO_EXTS:
        p = LOGOS_DIR / f"{base_key}{ext}"
        if p.exists():
            return p
    if not LOGOS_DIR.exists():
        return None
    target = base_key.lower()
    for p in LOGOS_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in LOGO_EXTS and p.stem.lower() == target:
            return p
    return None


def team_logo_uri(team_name: str) -> str:
    t = str(team_name or "").strip().lower()
    key = TEAM_KEY.get(t, "")
    if not key:
        return ""
    p = find_logo_file(key)
    if not p:
        return ""
    return _file_to_data_uri(str(p))


def race_logo_uri() -> str:
    if not RACE_LOGO_FILE.exists():
        return ""
    return _file_to_data_uri(str(RACE_LOGO_FILE))


def fmt_money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


def safe_first(df_: pd.DataFrame, col_name: str, default="—"):
    if col_name in df_.columns and not df_.empty:
        v = df_[col_name].iloc[0]
        return default if pd.isna(v) else v
    return default


def change_html(delta_abs: float, base: float) -> str:
    if base <= 0:
        return "<span class='chg_flat'>—</span>"

    # For very small base values, show only absolute change
    if base < MIN_BASE_FOR_PCT:
        if abs(delta_abs) < 1e-9:
            return "<span class='chg_flat'>0.00</span>"
        if delta_abs > 0:
            return f"<span class='chg_up'>+{delta_abs:,.2f}</span>"
        return f"<span class='chg_down'>-{abs(delta_abs):,.2f}</span>"

    pct = delta_abs / base

    # If ratio is too extreme, also show only absolute change
    if abs(pct) > MAX_PCT_FOR_DISPLAY:
        if abs(delta_abs) < 1e-9:
            return "<span class='chg_flat'>0.00</span>"
        if delta_abs > 0:
            return f"<span class='chg_up'>+{delta_abs:,.2f}</span>"
        return f"<span class='chg_down'>-{abs(delta_abs):,.2f}</span>"

    if abs(delta_abs) < 1e-9:
        return "<span class='chg_flat'>0.00 (0.0%)</span>"
    if delta_abs > 0:
        return f"<span class='chg_up'>+{delta_abs:,.2f} (+{pct*100:.1f}%)</span>"
    return f"<span class='chg_down'>-{abs(delta_abs):,.2f} (-{abs(pct)*100:.1f}%)</span>"


def spark_svg(values: List[float], up: bool, width: int = 170, height: int = 46) -> str:
    """Render a tiny SVG sparkline (no axes)."""
    if not values:
        return ""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return ""

    vmin = min(vals)
    vmax = max(vals)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0

    pad_x = 2
    pad_y = 2
    w = width - 2 * pad_x
    h = height - 2 * pad_y

    pts = []
    for i, v in enumerate(vals):
        x = pad_x + (i / (len(vals) - 1)) * w
        y = pad_y + (1 - ((v - vmin) / (vmax - vmin))) * h
        pts.append((x, y))

    poly = " ".join([f"{x:.1f},{y:.1f}" for x, y in pts])
    area = f"{poly} {pts[-1][0]:.1f},{pad_y+h:.1f} {pts[0][0]:.1f},{pad_y+h:.1f}"

    stroke = "#7CF6C2" if up else "#FF6B6B"
    fill = "rgba(124,246,194,0.18)" if up else "rgba(255,107,107,0.14)"

    area_path = "M " + " L ".join(area.split(" ")) + " Z"
    return (
        f"<svg class='spark' viewBox='0 0 {width} {height}' preserveAspectRatio='none'>"
        f"<path d='{area_path}' fill='{fill}'></path>"
        f"<polyline points='{poly}' fill='none' stroke='{stroke}' stroke-width='2.4' "
        f"stroke-linecap='round' stroke-linejoin='round'/>"
        f"</svg>"
    )


def build_spark_map(df_all: pd.DataFrame) -> Dict[str, List[float]]:
    """Map driver_key -> last SPARK_N fv values."""
    if df_all.empty:
        return {}
    d = df_all.copy()
    d["fv_dollars"] = pd.to_numeric(d.get("fv_dollars", 0.0), errors="coerce").fillna(0.0)
    d["driver_key"] = d["driver"].apply(normalize_driver_key)
    d = d.sort_values("ts_utc_dt")

    spark: Dict[str, List[float]] = {}
    for k, g in d.groupby("driver_key", sort=False):
        spark[k] = g["fv_dollars"].tail(SPARK_N).tolist()
    return spark


def build_lookup(df_snap: pd.DataFrame) -> Dict[str, float]:
    if df_snap.empty or "driver" not in df_snap.columns:
        return {}
    t = df_snap.copy()
    t["driver_key"] = t["driver"].apply(normalize_driver_key)
    t = t.drop_duplicates(subset=["driver_key"], keep="last")
    return dict(zip(t["driver_key"], t["fv_dollars"].astype(float)))


@st.cache_data(show_spinner=False, ttl=3600)
def logo_fit_transform(path_str: str) -> tuple[float, float, float]:
    """
    Compute a tasteful (scale, tx%, ty%) to reduce whitespace inside logo panels.

    - Uses alpha channel if present (best for PNG/WebP logos with transparent padding)
    - Falls back to trimming near-white background (for JPG/flat logos)
    - Caps scale to avoid aggressive cropping
    """
    p = Path(path_str)
    if not p.exists():
        return (1.0, 0.0, 0.0)

    img = Image.open(p).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[0], arr.shape[1]

    alpha = arr[..., 3]
    if alpha.max() > 0:
        mask = alpha > 8
    else:
        rgb = arr[..., :3].astype(np.int16)
        mask = np.any(rgb < 245, axis=-1)

    ys, xs = np.where(mask)
    if len(xs) < 25 or len(ys) < 25:
        return (1.0, 0.0, 0.0)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    cw = max(1, (x1 - x0 + 1))
    ch = max(1, (y1 - y0 + 1))

    scale_w = w / cw
    scale_h = h / ch
    scale = min(scale_w, scale_h)

    # keep it tasteful
    scale = float(max(1.0, min(scale, 1.55)))

    # recenter
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    img_cx = (w - 1) / 2.0
    img_cy = (h - 1) / 2.0

    dx = (img_cx - cx) / w
    dy = (img_cy - cy) / h

    tx = float(max(-18.0, min(dx * 100.0, 18.0)))
    ty = float(max(-18.0, min(dy * 100.0, 18.0)))

    return (scale, tx, ty)


def render_team_logo(team_name: str, wrap_class: str = "team_logo_wrap", img_class: str = "team_logo_img") -> str:
    """
    Render a team logo inside a white pill, auto-zooming to reduce whitespace.
    Falls back to text if logo not found.
    """
    t = str(team_name or "").strip()
    key = TEAM_KEY.get(t.lower(), "")
    p = find_logo_file(key) if key else None

    if not p:
        return f"<div class='{wrap_class}'><span style='color:#0b0d12;font-weight:900;'>{t or '—'}</span></div>"

    uri = _file_to_data_uri(str(p))
    scale, tx, ty = logo_fit_transform(str(p))

    return (
        f"<div class='{wrap_class}'>"
        f"<img class='{img_class}' src='{uri}' "
        f"style=\"transform: translate({tx:.2f}%, {ty:.2f}%) scale({scale:.3f});\"/>"
        f"</div>"
    )


def render_logo_by_key(logo_key: str, wrap_class: str, img_class: str) -> str:
    """
    Render a logo by file stem key (e.g., 'jhr'), with auto-fit.
    """
    p = find_logo_file(logo_key)
    if not p:
        return ""
    uri = _file_to_data_uri(str(p))
    scale, tx, ty = logo_fit_transform(str(p))
    return (
        f"<div class='{wrap_class}'>"
        f"<img class='{img_class}' src='{uri}' "
        f"style=\"transform: translate({tx:.2f}%, {ty:.2f}%) scale({scale:.3f});\"/>"
        f"</div>"
    )


def section_band_header(title: str, subtitle: str | None, alt: bool = False) -> None:
    """
    Renders a section band + header + divider as one single unsafe HTML call.
    """
    band_cls = "section_band section_band_alt" if alt else "section_band"
    sub_html = f"<div class='hdr_sub'>{subtitle}</div>" if subtitle else ""
    html = f"""
    <div class="{band_cls}">
      <div class="hdr_wrap">
        <div class="hdr_leftbar"></div>
        <div class="hdr_text">
          <div class="hdr_top">
            <div class="hdr_title">{title}</div>
          </div>
          {sub_html}
        </div>
      </div>
      <div class="section_divider"></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------- Page ----------------
st.set_page_config(page_title="DPC Market", layout="wide")

with st.sidebar:
    st.caption("Display")
    tv_safe = st.toggle("TV-safe mode", value=False, help="Higher contrast + bigger type for demo on screens.")
    st.caption("")

# --- STATIC CSS (NO f-string; braces are safe) ---
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none; }
    div[data-testid="stToolbar"] { display: none; }
    div[data-testid="stDecoration"] { display: none; }

    .block-container {
      padding-top: 0.45rem !important;
      padding-bottom: 2rem !important;
    }

    div[data-testid="stAppViewContainer"]{
      background:
        radial-gradient(circle at 18% 0%, rgba(103,245,181,0.10), transparent 52%),
        radial-gradient(circle at 85% 15%, rgba(103,245,181,0.06), transparent 58%),
        linear-gradient(180deg,#05060a,#0a0b10);
      color:#f2f4ff;
      position: relative;
    }
    div[data-testid="stAppViewContainer"] * { color:#f2f4ff; }

    div[data-testid="stAppViewContainer"] *{
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    div[data-testid="stAppViewContainer"]::before {
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
    }

    div[data-testid="stAppViewContainer"]::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      opacity: 0.10;
      background-image:
        repeating-linear-gradient(0deg, rgba(255,255,255,0.020), rgba(255,255,255,0.020) 1px, rgba(0,0,0,0.0) 2px, rgba(0,0,0,0.0) 4px),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.010), rgba(255,255,255,0.010) 1px, rgba(0,0,0,0.0) 2px, rgba(0,0,0,0.0) 6px);
    }

    .block-container, .block-container * { position: relative; z-index: 1; }

    div[data-testid="stSelectbox"] [data-baseweb="select"] > div{
      background:#f4f6f8 !important;
      border-radius:12px !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] span{ color:#0b0d12 !important; }
    div[data-testid="stSelectbox"] [data-baseweb="select"] div{ color:#0b0d12 !important; }
    div[role="listbox"]{ background:#f4f6f8 !important; }
    div[role="option"]{ color:#0b0d12 !important; }

    :root {
      --neon: rgba(124,246,194,0.95);
      --edge: rgba(255,255,255,0.10);
      --t12: 12px;
      --t14: 14px;
      --t16: 16px;
      --t20: 20px;
      --t28: 28px;
    }

    .section_band{
      margin-top: 14px;
      padding: 14px 14px 10px 14px;
      border-radius: 22px;
      border: 1px solid rgba(255,255,255,0.06);
      background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.014));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
    }
    .section_band_alt{
      background: linear-gradient(180deg, rgba(0,0,0,0.26), rgba(255,255,255,0.010));
    }
    .section_divider{
      height: 1px;
      margin: 12px 0 2px 0;
      background: linear-gradient(90deg,
        rgba(124,246,194,0.0),
        rgba(124,246,194,0.55),
        rgba(255,255,255,0.10),
        rgba(124,246,194,0.0)
      );
      opacity: .90;
    }

    .glass_card {
      border-radius: 18px;
      border: 1px solid var(--edge);
      background:
        radial-gradient(1200px 300px at 50% -120px, rgba(124,246,194,0.09), transparent 62%),
        linear-gradient(180deg, rgba(255,255,255,0.060), rgba(255,255,255,0.030));
      box-shadow:
        0 18px 50px rgba(0,0,0,.55),
        inset 0 1px 0 rgba(255,255,255,0.10),
        inset 0 -1px 0 rgba(0,0,0,0.40);
      position: relative;
    }
    .glass_card::before {
      content:"";
      position:absolute;
      inset:0;
      border-radius: 18px;
      pointer-events:none;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.0) 35%),
        radial-gradient(700px 180px at 25% 0%, rgba(255,255,255,0.08), transparent 60%);
      mix-blend-mode: screen;
      opacity: 0.55;
    }
    .glass_card::after {
      content:"";
      position:absolute;
      inset:0;
      border-radius: 18px;
      pointer-events:none;
      box-shadow: 0 0 0 1px rgba(124,246,194,0.06), 0 0 40px rgba(124,246,194,0.08);
      opacity: 0.40;
    }

    .accent { color:#7CF6C2 !important; }
    .muted { opacity: .78; }

    .neon_rule {
      height: 3px;
      border-radius: 999px;
      margin-top: 10px;
      background: linear-gradient(90deg, rgba(124,246,194,0.0), var(--neon), rgba(124,246,194,0.0));
      box-shadow: 0 10px 26px rgba(124,246,194,0.18);
    }

    .hdr_wrap { display:flex; gap: 12px; align-items:flex-start; margin-top: 8px; margin-bottom: 2px; }
    .hdr_leftbar {
      width: 6px;
      border-radius: 999px;
      background: linear-gradient(180deg, var(--neon), rgba(124,246,194,0.0));
      box-shadow: 0 10px 22px rgba(124,246,194,0.18);
      height: 40px;
      flex: 0 0 auto;
      margin-top: 3px;
    }
    .hdr_text { flex: 1 1 auto; }
    .hdr_top { display:flex; align-items:center; justify-content:space-between; gap: 12px; }
    .hdr_title { font-size: var(--t16); font-weight: 990; letter-spacing: .01em; }
    .hdr_sub {
      margin-top: 4px;
      font-size: var(--t12);
      font-weight: 850;
      opacity: .72;
      letter-spacing: .10em;
      text-transform: uppercase;
    }

    .live_dot {
      width: 10px; height: 10px; border-radius: 999px;
      background: #7CF6C2;
      box-shadow: 0 0 0 rgba(124,246,194,0.55);
      animation: livePulse 1.2s infinite;
    }
    @keyframes livePulse {
      0% { box-shadow: 0 0 0 0 rgba(124,246,194,0.55); }
      70% { box-shadow: 0 0 0 10px rgba(124,246,194,0.0); }
      100% { box-shadow: 0 0 0 0 rgba(124,246,194,0.0); }
    }
    .live_chip {
      display:inline-flex;
      align-items:center;
      gap: 10px;
      padding: 7px 11px;
      border-radius: 999px;
      border: 1px solid rgba(124,246,194,0.22);
      background: rgba(124,246,194,0.09);
      font-size: var(--t12);
      font-weight: 980;
      letter-spacing: .10em;
      text-transform: uppercase;
      white-space: nowrap;
    }

    .ticker {
      margin-top: 10px;
      border-radius: 18px;
      border: 1px solid rgba(124,246,194,0.14);
      background: linear-gradient(180deg, rgba(0,0,0,0.55), rgba(0,0,0,0.22));
      box-shadow: 0 18px 44px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.08);
      overflow: hidden;
      padding: 10px 12px;
      position: relative;
    }
    .ticker_track {
      display: inline-block;
      white-space: nowrap;
      animation: scrollLeft 24s linear infinite;
      font-size: var(--t14);
      opacity: .95;
      font-weight: 900;
      letter-spacing: .02em;
      text-shadow: 0 8px 22px rgba(0,0,0,0.55);
    }
    @keyframes scrollLeft { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
    .ticker_sep { opacity:.55; padding: 0 14px; }

    .header_logo_wrap, .race_logo_wrap{
      background: rgba(244,246,248,0.93);
      border-radius: 16px;
      padding: 10px 16px;
      display:flex;
      align-items:center;
      justify-content:center;
      box-shadow: 0 10px 26px rgba(0,0,0,0.35);
      overflow: hidden;
    }
    .header_logo_img{ height: 92px; width:auto; display:block; transform-origin: center center; }
    .race_logo_img{ height: 76px; width:auto; display:block; transform-origin: center center; }

    .snap_strip{
      margin-top: 10px;
      padding: 10px 14px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.10);
      background:
        radial-gradient(1200px 220px at 50% -120px, rgba(124,246,194,0.10), transparent 65%),
        linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.025));
      box-shadow: 0 18px 50px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.10);
      display:flex;
      align-items:center;
      gap:12px;
      flex-wrap: wrap;
      position: relative;
    }
    .snap_item{
      display:flex;
      align-items:baseline;
      gap:8px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(0,0,0,0.22);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .snap_k{ font-size: var(--t12); opacity: .75; font-weight: 850; }
    .snap_v{ font-size: var(--t14); font-weight: 950; opacity: .95; white-space: nowrap; }
    .pill_green{ background: rgba(124,246,194,0.12) !important; border-color: rgba(124,246,194,0.22) !important; }
    .pill_yellow{ background: rgba(255,220,120,0.12) !important; border-color: rgba(255,220,120,0.22) !important; }

    .team_logo_wrap{
      background:#f4f6f8;
      border-radius: 12px;
      padding: 6px 12px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      height: 44px;
      min-width: 140px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.35);
      overflow: hidden;
    }
    .team_logo_img{
      max-height: 30px;
      max-width: 120px;
      width: auto;
      height: auto;
      object-fit: contain;
      display:block;
      transform-origin: center center;
    }

    .panel_row { margin-top: 10px; display:flex; gap:14px; flex-wrap: wrap; }
    .panel {
      flex: 1 1 320px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.10);
      background:
        radial-gradient(900px 220px at 35% -120px, rgba(124,246,194,0.09), transparent 65%),
        linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.030));
      box-shadow: 0 18px 50px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.10);
      padding: 14px 14px;
      position: relative;
    }
    .panel::before{
      content:"";
      position:absolute;
      inset:0;
      border-radius:16px;
      pointer-events:none;
      background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.0) 40%);
      opacity: 0.50;
    }
    .panel_h { display:flex; align-items:center; justify-content:space-between; font-size: var(--t16); font-weight: 990; letter-spacing: .01em; margin-bottom: 8px; }
    .panel_sub { font-size: var(--t12); opacity: .72; font-weight: 850; letter-spacing: .10em; text-transform: uppercase; }

    .spark { width: 170px; height: 46px; display:block; opacity: .98; }
    .spark_small { width: 150px; height: 40px; display:block; opacity: .98; margin-top: 6px; }

    .top3_wrap{ margin-top: 10px; display:flex; gap:12px; flex-wrap: wrap; }
    .top3_card{
      flex: 1 1 260px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.10);
      background:
        radial-gradient(700px 220px at 35% -120px, rgba(124,246,194,0.10), transparent 70%),
        linear-gradient(180deg, rgba(255,255,255,0.060), rgba(255,255,255,0.030));
      box-shadow: 0 18px 50px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.10);
      padding: 12px 14px;
      min-width: 240px;
      position: relative;
      overflow: hidden;
    }
    .top3_card::before{
      content:"";
      position:absolute;
      inset:0;
      border-radius:16px;
      pointer-events:none;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.0) 40%),
        radial-gradient(700px 200px at 15% 0%, rgba(255,255,255,0.07), transparent 60%);
      opacity: 0.55;
    }
    .top3_leader{
      border-color: rgba(124,246,194,0.26);
      background:
        radial-gradient(900px 260px at 35% -140px, rgba(124,246,194,0.14), transparent 70%),
        linear-gradient(180deg, rgba(124,246,194,0.08), rgba(255,255,255,0.030));
      box-shadow:
        0 18px 50px rgba(0,0,0,0.55),
        0 0 50px rgba(124,246,194,0.12),
        inset 0 1px 0 rgba(255,255,255,0.10);
    }
    .top3_top{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
    .top3_left{ display:flex; align-items:center; gap:10px; min-width:0; }
    .flag_img{ width: 28px; height: 20px; border-radius: 6px; object-fit: cover; box-shadow: 0 6px 16px rgba(0,0,0,0.35); flex:0 0 auto; }
    .top3_rank{ font-size: var(--t12); font-weight: 980; letter-spacing: .10em; opacity: .92; flex:0 0 auto; }
    .top3_name{ font-size: var(--t16); font-weight: 990; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.1; }
    .top3_price{ text-align:right; flex:0 0 auto; }
    .top3_fv{ font-size: var(--t20); font-weight: 995; letter-spacing: .01em; }
    .top3_chg, .top3_open{ font-size: var(--t12); font-weight: 900; margin-top: 2px; opacity: .95; white-space: nowrap; }
    .top3_bottom{ margin-top: 10px; display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .top3_pos{ font-size: var(--t14); opacity: .92; white-space: nowrap; }
    .top3_meta{ margin-top: 10px; display:flex; align-items:center; justify-content:space-between; gap:12px; }

    .m_row{
      display:flex; align-items:center; justify-content:space-between;
      padding: 10px 10px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(0,0,0,0.18);
      margin-top: 10px;
      gap:12px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .m_left{ display:flex; align-items:center; gap:10px; min-width:0; }
    .m_name{ font-size: var(--t14); font-weight: 950; white-space: nowrap; overflow:hidden; text-overflow: ellipsis; }
    .m_right{ text-align:right; flex:0 0 auto; }
    .m_delta{ font-size: var(--t14); font-weight: 980; white-space: nowrap; }

    .driver_card{
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.10);
      background:
        radial-gradient(900px 220px at 35% -140px, rgba(124,246,194,0.08), transparent 70%),
        linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.028));
      box-shadow: 0 18px 50px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.10);
      padding: 12px 14px;
      margin: 9px 0;
      position: relative;
      overflow: hidden;
    }
    .driver_card::before{
      content:"";
      position:absolute;
      inset:0;
      border-radius:16px;
      pointer-events:none;
      background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.0) 42%);
      opacity: 0.50;
    }
    .leader_card{
      border-color: rgba(124,246,194,0.22);
      background:
        radial-gradient(900px 260px at 35% -160px, rgba(124,246,194,0.12), transparent 70%),
        linear-gradient(180deg, rgba(124,246,194,0.06), rgba(255,255,255,0.028));
      box-shadow:
        0 18px 50px rgba(0,0,0,0.55),
        0 0 45px rgba(124,246,194,0.10),
        inset 0 1px 0 rgba(255,255,255,0.10);
    }
    @keyframes pulseUp{
      0%{ box-shadow: 0 18px 50px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.10); }
      35%{ box-shadow: 0 18px 58px rgba(124,246,194,0.22), inset 0 1px 0 rgba(255,255,255,0.10); }
      100%{ box-shadow: 0 18px 50px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.10); }
    }
    @keyframes pulseDown{
      0%{ box-shadow: 0 18px 50px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.10); }
      35%{ box-shadow: 0 18px 58px rgba(255,107,107,0.18), inset 0 1px 0 rgba(255,255,255,0.10); }
      100%{ box-shadow: 0 18px 50px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.10); }
    }
    .pulse_up{ animation: pulseUp 0.9s ease-out 1; }
    .pulse_down{ animation: pulseDown 0.9s ease-out 1; }

    .driver_line1{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
    .driver_left{ display:flex; align-items:center; gap:10px; min-width: 0; }
    .rank_badge{
      font-size: var(--t12);
      font-weight: 980;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(0,0,0,0.25);
      opacity: .95;
      flex: 0 0 auto;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .driver_name{
      font-size: var(--t16);
      font-weight: 990;
      line-height: 1.12;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .price_block{ text-align:right; flex:0 0 auto; display:flex; flex-direction:column; align-items:flex-end; gap:4px; }
    .driver_fv{ font-size: var(--t20); font-weight: 995; white-space: nowrap; }
    .driver_chg{ font-size: var(--t12); font-weight: 900; white-space: nowrap; opacity: .95; }
    .driver_open{ font-size: var(--t12); font-weight: 850; opacity: .80; white-space: nowrap; }
    .driver_spark_wrap{ margin-top: 2px; }

    .chg_up{ color:#7CF6C2 !important; }
    .chg_down{ color:#FF6B6B !important; }
    .chg_flat{ opacity: .72; }

    .driver_line2{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top: 10px; }
    .driver_meta{ font-size: var(--t14); opacity: .92; white-space: nowrap; flex: 0 0 auto; }
    .delta_up{ color:#7CF6C2 !important; font-weight:980; }
    .delta_down{ color:#FF6B6B !important; font-weight:980; }

    .detail_wrap { margin-top: 10px; display:flex; gap:12px; flex-wrap: wrap; }
    .detail_tile {
      flex: 1 1 190px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.10);
      background:
        radial-gradient(700px 200px at 35% -120px, rgba(124,246,194,0.08), transparent 70%),
        linear-gradient(180deg, rgba(255,255,255,0.050), rgba(255,255,255,0.028));
      box-shadow: 0 18px 50px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.10);
      padding: 12px 14px;
      position: relative;
      overflow: hidden;
    }
    .detail_tile::before{
      content:"";
      position:absolute;
      inset:0;
      border-radius:16px;
      pointer-events:none;
      background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.0) 45%);
      opacity: 0.45;
    }
    .detail_k{ font-size: var(--t12); letter-spacing: .10em; text-transform: uppercase; opacity: .75; font-weight: 900; }
    .detail_v{ margin-top: 6px; font-size: var(--t20); font-weight: 990; white-space: nowrap; }

    .history_row{ display:flex; align-items:center; gap:14px; margin-top: 8px; margin-bottom: 4px; }
    .history_flag{
      width: 62px;
      height: 46px;
      border-radius: 12px;
      object-fit: cover;
      box-shadow: 0 10px 24px rgba(0,0,0,0.35);
      flex: 0 0 auto;
    }
    .history_name{
      font-size: var(--t28);
      font-weight: 995;
      line-height: 1.05;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      flex: 1 1 auto;
    }

    .driver_card, .top3_card, .panel, .detail_tile {
      transition: transform .12s ease, border-color .12s ease, box-shadow .12s ease;
    }
    .driver_card:hover, .top3_card:hover, .panel:hover, .detail_tile:hover {
      transform: translateY(-1px);
      border-color: rgba(124,246,194,0.18);
      box-shadow: 0 22px 58px rgba(0,0,0,0.62), inset 0 1px 0 rgba(255,255,255,0.10);
    }

    @media (max-width: 768px){
      .block-container { padding-top: 0.35rem !important; }
      .header_logo_img{ height: 72px; }
      .race_logo_img{ height: 60px; }
      .snap_strip{ padding: 10px 12px; }
      .panel{ min-width: 100%; }
      .top3_card{ min-width: 100%; }
      .spark{ width: 150px; height: 44px; }
      .driver_spark_wrap .spark{ width: 150px; height: 40px; }
      .history_name{ font-size: 20px; }
      .hdr_leftbar { height: 34px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- DYNAMIC CSS (tiny f-string) ---
gap_px = 10 if tv_safe else 8
t12 = 13 if tv_safe else 12
t14 = 15 if tv_safe else 14
t16 = 18 if tv_safe else 16
t20 = 23 if tv_safe else 20
t28 = 32 if tv_safe else 28
logo_h = 96 if tv_safe else 92
race_h = 80 if tv_safe else 76
team_h = 48 if tv_safe else 44
team_img_h = 34 if tv_safe else 30
hist_flag_w = 70 if tv_safe else 62
hist_flag_h = 52 if tv_safe else 46

st.markdown(
    f"""
    <style>
      div[data-testid="stVerticalBlock"] > div {{ gap: {gap_px}px; }}
      :root {{
        --t12: {t12}px;
        --t14: {t14}px;
        --t16: {t16}px;
        --t20: {t20}px;
        --t28: {t28}px;
      }}
      .header_logo_img {{ height: {logo_h}px; }}
      .race_logo_img {{ height: {race_h}px; }}
      .team_logo_wrap {{ height: {team_h}px; }}
      .team_logo_img {{ max-height: {team_img_h}px; }}
      .history_flag {{ width: {hist_flag_w}px; height: {hist_flag_h}px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

conn = connect_db()
df = read_ticks(conn)

if df.empty:
    st.warning("No published marks yet. Admin must publish at least one snapshot.")
    st.stop()

df["ts_utc_dt"] = pd.to_datetime(df.get("ts_utc", ""), utc=True, errors="coerce")
df = df.dropna(subset=["ts_utc_dt"])
if df.empty:
    st.warning("No valid timestamps found in ticks.")
    st.stop()

# Latest snapshot
latest_ts = df["ts_utc_dt"].max()
latest = df[df["ts_utc_dt"] == latest_ts].copy()

# Previous snapshot
prev_ts = df.loc[df["ts_utc_dt"] < latest_ts, "ts_utc_dt"].max()
prev = df[df["ts_utc_dt"] == prev_ts].copy() if pd.notna(prev_ts) else pd.DataFrame()

# Session open snapshot (earliest tick)
open_ts = df["ts_utc_dt"].min()
open_df = df[df["ts_utc_dt"] == open_ts].copy()

# Numeric FV
df["fv_dollars"] = pd.to_numeric(df.get("fv_dollars", 0.0), errors="coerce").fillna(0.0)
latest["fv_dollars"] = pd.to_numeric(latest.get("fv_dollars", 0.0), errors="coerce").fillna(0.0)
if not prev.empty:
    prev["fv_dollars"] = pd.to_numeric(prev.get("fv_dollars", 0.0), errors="coerce").fillna(0.0)
if not open_df.empty:
    open_df["fv_dollars"] = pd.to_numeric(open_df.get("fv_dollars", 0.0), errors="coerce").fillna(0.0)

prev_fv_by_driver = build_lookup(prev)
open_fv_by_driver = build_lookup(open_df)
spark_map = build_spark_map(df)

# ---------------- Header ----------------
jhr_html = render_logo_by_key("jhr", "header_logo_wrap", "header_logo_img") or "<div class='header_logo_wrap'><b style='color:#0b0d12;'>JHR</b></div>"

race_uri = race_logo_uri()
if race_uri:
    # Try to auto-fit race logo too if it exists locally
    if RACE_LOGO_FILE.exists():
        scale, tx, ty = logo_fit_transform(str(RACE_LOGO_FILE))
        race_html = (
            "<div class='race_logo_wrap'>"
            f"<img class='race_logo_img' src='{race_uri}' "
            f"style=\"transform: translate({tx:.2f}%, {ty:.2f}%) scale({scale:.3f});\"/>"
            "</div>"
        )
    else:
        race_html = f"<div class='race_logo_wrap'><img class='race_logo_img' src='{race_uri}'/></div>"
else:
    race_html = ""

st.markdown(
    f"<div class='glass_card' style='padding:{16 if tv_safe else 14}px {20 if tv_safe else 18}px; display:flex;align-items:center;gap:16px;'>"
    f"<div style='display:flex;align-items:center;justify-content:center;'>{jhr_html}</div>"
    f"<div style='min-width:0;'>"
    f"  <div style='font-size:24px;font-weight:995;line-height:1.10;letter-spacing:.01em;'>{APP_TITLE}</div>"
    f"  <div class='muted' style='font-size:13px;margin-top:3px;'>{SUBTITLE}</div>"
    f"  <div class='neon_rule'></div>"
    f"</div>"
    f"<div style='flex:1;'></div>"
    f"<div style='display:flex;align-items:center;justify-content:center;'>{race_html}</div>"
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
                if pct <= MAX_PCT_FOR_DISPLAY:
                    pct_moves.append(pct)
    if pct_moves:
        avg_abs_pct = sum(pct_moves) / len(pct_moves)

flag_lower = flag_state.lower()
pill_class = "pill_yellow" if flag_lower.startswith("yellow") else "pill_green"

strip_parts = []
strip_parts.append("<div class='snap_strip'>")
strip_parts.append("<div class='live_chip'><span class='live_dot'></span> LIVE</div>")
strip_parts.append(f"<div class='snap_item'><div class='snap_k'>Lap</div><div class='snap_v'>{lap}/{total_laps if total_laps else '—'}</div></div>")
strip_parts.append(f"<div class='snap_item {pill_class}'><div class='snap_k'>Flag</div><div class='snap_v'>{flag_state}</div></div>")
strip_parts.append(f"<div class='snap_item'><div class='snap_k'>Last update</div><div class='snap_v'>{last_update}</div></div>")
strip_parts.append(f"<div class='snap_item'><div class='snap_k'>Total market</div><div class='snap_v'>{fmt_money(total_mkt)}</div></div>")
if avg_abs_pct is not None:
    strip_parts.append(f"<div class='snap_item'><div class='snap_k'>Avg move</div><div class='snap_v'>{avg_abs_pct*100:.1f}%</div></div>")
if prev_fv_by_driver:
    strip_parts.append(
        f"<div class='snap_item'><div class='snap_k'>Breadth</div>"
        f"<div class='snap_v'><span class='chg_up'>{adv}↑</span> / <span class='chg_down'>{dec}↓</span></div></div>"
    )
strip_parts.append("</div>")
st.markdown("".join(strip_parts), unsafe_allow_html=True)

headline_items = [f"Total market {fmt_money(total_mkt)}"]
if prev_fv_by_driver:
    mover_sign = "+" if biggest_mover_delta >= 0 else "−"
    headline_items.append(f"Biggest mover: {biggest_mover_name} {mover_sign}{abs(biggest_mover_delta):,.2f}")
    headline_items.append(f"Market breadth: {adv} up / {dec} down")
if avg_abs_pct is not None:
    headline_items.append(f"Avg move (capped): {avg_abs_pct*100:.1f}%")
headline_items.append(f"Last update {last_update}")
ticker_text = "  ".join([f"{x}<span class='ticker_sep'>•</span>" for x in headline_items])
st.markdown(f"<div class='ticker'><div class='ticker_track'>{ticker_text}</div></div>", unsafe_allow_html=True)

with st.expander("Controls", expanded=False):
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_seconds = st.slider("Refresh every (seconds)", 5, 60, 10, 5, disabled=not auto_refresh)

# ---------------- Leaders & Movers ----------------
section_band_header("Leaders & Movers", subtitle="LIVE PRICING", alt=False)

latest_sorted = latest.sort_values("fv_dollars", ascending=False).reset_index(drop=True)

# Movers compute
deltas = []
if prev_fv_by_driver:
    for _, r in latest_sorted.iterrows():
        key = normalize_driver_key(r.get("driver", ""))
        nowv = float(r.get("fv_dollars", 0.0) or 0.0)
        prevv = float(prev_fv_by_driver.get(key, 0.0) or 0.0)
        if prevv > 0:
            deltas.append((key, r, nowv - prevv, prevv))

top_up = sorted([x for x in deltas if x[2] > 0], key=lambda z: z[2], reverse=True)[:3]
top_dn = sorted([x for x in deltas if x[2] < 0], key=lambda z: z[2])[:3]

st.markdown("<div class='panel_row'>", unsafe_allow_html=True)

# Top 3 panel
st.markdown(
    "<div class='panel'>"
    "<div class='panel_h'><span>Top Contracts</span><span class='panel_sub'>Leaders</span></div>",
    unsafe_allow_html=True,
)

top3 = latest_sorted.head(3).copy()
top_html = ["<div class='top3_wrap'>"]

for i, r in top3.iterrows():
    rank = i + 1
    driver_raw = r.get("driver", "")
    driver_key = normalize_driver_key(driver_raw)
    driver = clean_driver_name(driver_raw)

    team = str(r.get("team", "") or "")
    country = str(r.get("country", "") or "")
    fv = float(r.get("fv_dollars", 0.0) or 0.0)

    grid = int(pd.to_numeric(r.get("grid_pos", 0), errors="coerce") or 0)
    pos = int(pd.to_numeric(r.get("position", 0), errors="coerce") or 0)
    pos_delta = grid - pos

    pos_delta_html = ""
    if pos_delta > 0:
        pos_delta_html = f" <span class='delta_up'>▲ {pos_delta}</span>"
    elif pos_delta < 0:
        pos_delta_html = f" <span class='delta_down'>▼ {abs(pos_delta)}</span>"

    mark_line = "<span class='chg_flat'>—</span>"
    if prev_fv_by_driver:
        prev_val = float(prev_fv_by_driver.get(driver_key, 0.0) or 0.0)
        if prev_val > 0:
            mark_line = change_html(fv - prev_val, prev_val)

    open_line = "<span class='chg_flat'>—</span>"
    open_delta_abs = 0.0
    if open_fv_by_driver:
        open_val = float(open_fv_by_driver.get(driver_key, 0.0) or 0.0)
        if open_val > 0:
            open_delta_abs = fv - open_val
            open_line = change_html(open_delta_abs, open_val)

    flag_img = flag_url(country)
    flag_html = f"<img class='flag_img' src='{flag_img}'/>" if flag_img else ""

    team_html = render_team_logo(team, "team_logo_wrap", "team_logo_img")

    sp = spark_svg(spark_map.get(driver_key, []), up=(open_delta_abs >= 0))
    leader_cls = "top3_card top3_leader" if rank == 1 else "top3_card"

    top_html.append(f"<div class='{leader_cls}'>")
    top_html.append("<div class='top3_top'>")
    top_html.append("<div class='top3_left'>")
    top_html.append(f"<div class='top3_rank'>#{rank}</div>")
    top_html.append(flag_html)
    top_html.append(f"<div class='top3_name'>{driver}</div>")
    top_html.append("</div>")
    top_html.append("<div class='top3_price'>")
    top_html.append(f"<div class='top3_fv'>{fmt_money(fv)}</div>")
    top_html.append(f"<div class='top3_chg'>Mark: {mark_line}</div>")
    top_html.append(f"<div class='top3_open' style='opacity:.85;'>Open: {open_line}</div>")
    top_html.append("</div>")
    top_html.append("</div>")

    top_html.append("<div class='top3_bottom'>")
    top_html.append(sp if sp else "<div style='opacity:.65;font-size:12px;'>No sparkline</div>")
    top_html.append(f"<div class='top3_pos'>Start <b>{grid}</b> → Now <b class='accent'>{pos}</b>{pos_delta_html}</div>")
    top_html.append("</div>")

    top_html.append("<div class='top3_meta'>")
    top_html.append(team_html)
    top_html.append("<div style='flex:1;'></div>")
    top_html.append("</div>")
    top_html.append("</div>")

top_html.append("</div>")
st.markdown("".join(top_html), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Movers panel
st.markdown(
    "<div class='panel'>"
    "<div class='panel_h'><span>Biggest Movers</span><span class='panel_sub'>Since last mark</span></div>",
    unsafe_allow_html=True,
)


def mover_row(r, delta_abs: float, base_prev: float) -> str:
    driver_key = normalize_driver_key(r.get("driver", ""))
    name = clean_driver_name(r.get("driver", ""))
    country = str(r.get("country", "") or "")
    flag_img = flag_url(country)
    flag_html = f"<img class='flag_img' src='{flag_img}'/>" if flag_img else ""
    nowv = float(r.get("fv_dollars", 0.0) or 0.0)

    sp = spark_svg(spark_map.get(driver_key, []), up=(delta_abs >= 0), width=150, height=40)
    sp = sp.replace("class='spark'", "class='spark_small'")
    delta_txt = change_html(delta_abs, base_prev)

    return (
        "<div class='m_row'>"
        f"<div class='m_left'>{flag_html}<div style='min-width:0;'><div class='m_name'>{name}</div>"
        f"<div style='opacity:.70;font-size:12px;margin-top:2px;'>{fmt_money(nowv)}</div></div></div>"
        f"<div class='m_right'><div class='m_delta'>{delta_txt}</div>{sp}</div>"
        "</div>"
    )


rows_html = []
for item in top_up[:2]:
    _, rr, d_abs, b = item
    rows_html.append(mover_row(rr, d_abs, b))
for item in top_dn[:2]:
    _, rr, d_abs, b = item
    rows_html.append(mover_row(rr, d_abs, b))

if not rows_html:
    rows_html.append("<div style='opacity:.75;font-size:14px;'>No mover data yet (publish at least 2 snapshots).</div>")

st.markdown("".join(rows_html), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # close movers panel
st.markdown("</div>", unsafe_allow_html=True)  # close panel_row

# ---------------- Live Contract Board ----------------
section_band_header("Live Contract Board", subtitle="ALL DRIVERS", alt=True)

latest_sorted = latest.sort_values("fv_dollars", ascending=False)

for idx, (_, r) in enumerate(latest_sorted.iterrows(), start=1):
    driver_raw = r.get("driver", "")
    driver_key = normalize_driver_key(driver_raw)
    driver = clean_driver_name(driver_raw)

    team = str(r.get("team", "") or "")
    country = str(r.get("country", "") or "")
    fv = float(r.get("fv_dollars", 0.0) or 0.0)

    grid = int(pd.to_numeric(r.get("grid_pos", 0), errors="coerce") or 0)
    pos = int(pd.to_numeric(r.get("position", 0), errors="coerce") or 0)
    pos_delta = grid - pos

    pos_delta_html = ""
    if pos_delta > 0:
        pos_delta_html = f"&nbsp; <span class='delta_up'>▲ {pos_delta}</span>"
    elif pos_delta < 0:
        pos_delta_html = f"&nbsp; <span class='delta_down'>▼ {abs(pos_delta)}</span>"

    chg_html = "<span class='chg_flat'>—</span>"
    delta_abs_for_pulse = 0.0
    if prev_fv_by_driver:
        fv_prev = float(prev_fv_by_driver.get(driver_key, 0.0) or 0.0)
        if fv_prev > 0:
            delta_abs_for_pulse = fv - fv_prev
            chg_html = change_html(delta_abs_for_pulse, fv_prev)

    open_html = ""
    open_delta_abs = 0.0
    if open_fv_by_driver:
        fv_open = float(open_fv_by_driver.get(driver_key, 0.0) or 0.0)
        if fv_open > 0:
            open_delta_abs = fv - fv_open
            open_line = change_html(open_delta_abs, fv_open)
            open_html = f"<div class='driver_open'>Since open: {open_line}</div>"

    direction_up = True
    if open_fv_by_driver and driver_key in open_fv_by_driver:
        direction_up = open_delta_abs >= 0
    elif prev_fv_by_driver and driver_key in prev_fv_by_driver:
        direction_up = delta_abs_for_pulse >= 0

    sp = spark_svg(spark_map.get(driver_key, []), up=direction_up)
    spark_html = f"<div class='driver_spark_wrap'>{sp}</div>" if sp else ""

    flag_img = flag_url(country)
    flag_html = f"<img class='flag_img' src='{flag_img}'/>" if flag_img else ""

    team_html = render_team_logo(team, "team_logo_wrap", "team_logo_img")

    base_class = "driver_card leader_card" if idx == 1 else "driver_card"
    pulse_class = ""
    if abs(delta_abs_for_pulse) >= PULSE_ABS_DOLLAR_THRESHOLD:
        pulse_class = " pulse_up" if delta_abs_for_pulse > 0 else " pulse_down"

    st.markdown(
        f"<div class='{base_class}{pulse_class}'>"
        f"<div class='driver_line1'>"
        f"<div class='driver_left'>"
        f"<span class='rank_badge'>#{idx}</span>"
        f"{flag_html}"
        f"<div class='driver_name'>{driver}</div>"
        f"</div>"
        f"<div class='price_block'>"
        f"<div class='driver_fv'>{fmt_money(fv)}</div>"
        f"<div class='driver_chg'>{chg_html}</div>"
        f"{open_html}"
        f"{spark_html}"
        f"</div>"
        f"</div>"
        f"<div class='driver_line2'>"
        f"<div>{team_html}</div>"
        f"<div class='driver_meta'>Start: <b>{grid}</b> → Now: <b class='accent'>{pos}</b>{pos_delta_html}</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ---------------- Fair Value History ----------------
section_band_header("Fair Value History", subtitle="OPEN / HIGH / LOW + LAST", alt=False)

need_cols = {"driver", "team", "country"}
driver_info = latest_sorted[list(need_cols)].dropna().drop_duplicates() if need_cols.issubset(latest_sorted.columns) else pd.DataFrame()
if driver_info.empty and need_cols.issubset(df.columns):
    driver_info = df[list(need_cols)].dropna().drop_duplicates()

if driver_info.empty:
    st.info("No driver metadata available yet.")
    st.stop()

options = []
option_to_driver = {}
for _, rr in driver_info.iterrows():
    dname = str(rr["driver"])
    tname = str(rr.get("team", ""))
    label = f"{clean_driver_name(dname)} — {tname}".strip()
    options.append(label)
    option_to_driver[label] = dname
options = sorted(options)

sel_label = st.selectbox("Select driver", options)
sel_driver = option_to_driver[sel_label]
sel_key = normalize_driver_key(sel_driver)

row_match = driver_info[driver_info["driver"] == sel_driver]
sel_team = str(row_match["team"].iloc[0]) if not row_match.empty else ""
sel_country = str(row_match["country"].iloc[0]) if not row_match.empty else ""

team_html = render_team_logo(sel_team, "team_logo_wrap", "team_logo_img")
flag_img = flag_url(sel_country)
flag_html = f"<img class='history_flag' src='{flag_img}'/>" if flag_img else ""

latest_row = latest_sorted[latest_sorted["driver"].apply(normalize_driver_key) == sel_key]
fv_now = float(latest_row["fv_dollars"].iloc[0]) if not latest_row.empty else 0.0

rank_now = (
    int(latest_sorted.reset_index(drop=True).index[latest_sorted["driver"].apply(normalize_driver_key) == sel_key][0] + 1)
    if not latest_sorted.empty and any(latest_sorted["driver"].apply(normalize_driver_key) == sel_key)
    else 0
)

grid_now = int(pd.to_numeric(latest_row.get("grid_pos", pd.Series([0])).iloc[0], errors="coerce") or 0) if not latest_row.empty else 0
pos_now = int(pd.to_numeric(latest_row.get("position", pd.Series([0])).iloc[0], errors="coerce") or 0) if not latest_row.empty else 0
pos_delta = grid_now - pos_now

since_mark = "—"
if prev_fv_by_driver and sel_key in prev_fv_by_driver:
    base = float(prev_fv_by_driver.get(sel_key, 0.0) or 0.0)
    if base > 0:
        since_mark = change_html(fv_now - base, base)

since_open = "—"
if open_fv_by_driver and sel_key in open_fv_by_driver:
    base = float(open_fv_by_driver.get(sel_key, 0.0) or 0.0)
    if base > 0:
        since_open = change_html(fv_now - base, base)

pos_txt = f"Start {grid_now} → Now {pos_now}"
if pos_delta > 0:
    pos_txt += f" ▲{pos_delta}"
elif pos_delta < 0:
    pos_txt += f" ▼{abs(pos_delta)}"

st.markdown(
    f"<div class='history_row'>{flag_html}"
    f"<div class='history_name'>{clean_driver_name(sel_driver)}</div>"
    f"<div>{team_html}</div></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='detail_wrap'>"
    f"<div class='detail_tile'><div class='detail_k'>Current FV</div><div class='detail_v'>{fmt_money(fv_now)}</div></div>"
    f"<div class='detail_tile'><div class='detail_k'>Rank</div><div class='detail_v'>#{rank_now if rank_now else '—'}</div></div>"
    f"<div class='detail_tile'><div class='detail_k'>Since Open</div><div class='detail_v'>{since_open}</div></div>"
    f"<div class='detail_tile'><div class='detail_k'>Since Mark</div><div class='detail_v'>{since_mark}</div></div>"
    f"<div class='detail_tile'><div class='detail_k'>Position</div><div class='detail_v'>{pos_txt}</div></div>"
    "</div>",
    unsafe_allow_html=True,
)

hist = df[df["driver"].apply(normalize_driver_key) == sel_key].copy()
hist = hist.sort_values("ts_utc_dt")
hist["fv_dollars"] = pd.to_numeric(hist.get("fv_dollars", 0.0), errors="coerce").fillna(0.0)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=hist["ts_utc_dt"],
        y=hist["fv_dollars"],
        mode="lines",
        line=dict(color="#7CF6C2", width=3 if tv_safe else 2.8),
        fill="tozeroy",
        fillcolor="rgba(124,246,194,0.10)",
        hovertemplate="%{x|%H:%M:%S UTC}<br>$%{y:,.2f}<extra></extra>",
        name="FV",
    )
)

if not hist.empty:
    y_open = float(hist["fv_dollars"].iloc[0])
    y_high = float(hist["fv_dollars"].max())
    y_low = float(hist["fv_dollars"].min())

    for y, label, dash, alpha in [
        (y_open, "OPEN", "dot", 0.55),
        (y_high, "HIGH", "dash", 0.40),
        (y_low, "LOW", "dash", 0.40),
    ]:
        fig.add_hline(
            y=y,
            line_width=2 if tv_safe else 1.6,
            line_dash=dash,
            line_color=f"rgba(243,246,255,{alpha})",
            opacity=1.0,
        )
        fig.add_annotation(
            x=1.0,
            xref="paper",
            xanchor="right",
            y=y,
            yref="y",
            text=f"{label}  {fmt_money(y)}",
            showarrow=False,
            font=dict(size=12 if tv_safe else 11, color="rgba(243,246,255,0.85)"),
            bgcolor="rgba(0,0,0,0.32)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            borderpad=4,
        )

if not hist.empty:
    x_open = hist["ts_utc_dt"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=[x_open],
            y=[float(hist["fv_dollars"].iloc[0])],
            mode="markers",
            marker=dict(
                size=10 if tv_safe else 9,
                color="rgba(243,246,255,0.88)",
                line=dict(width=1, color="rgba(0,0,0,0.35)"),
            ),
            hovertemplate="Open<br>%{x|%H:%M:%S UTC}<br>$%{y:,.2f}<extra></extra>",
            name="Open",
        )
    )

if not hist.empty:
    x_last = hist["ts_utc_dt"].iloc[-1]
    y_last = float(hist["fv_dollars"].iloc[-1])
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
    height=440 if tv_safe else 410,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.20)",
    font=dict(color="#F3F6FF", size=16 if tv_safe else 15),
    margin=dict(l=18, r=18, t=26, b=16),
    showlegend=False,
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.14)",
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