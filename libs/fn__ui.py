"""
libs/fn__ui.py
==============
UI helpers: global CSS injection, dividers, styled regions,
DataFrame stylisers, and the EPW location map.

Number range: 203–211  (UI series)

Absorbs:
  - libs/shared.py
  - UI helper functions from libs/fn__libraries.py

Function index
--------------
  f203__inject_global_css          – inject fonts + all global CSS
  f204__custom_divider             – configurable <hr> divider
  f205__custom_hr                  – legacy compact <hr>
  f206__tight_hr_spacing           – legacy tight-spacing CSS injection
  f207__custom_filled_region       – styled filled text region
  f208__style_df                   – gradient-background DataFrame Styler
  f209__highlight_odd_even_cells   – per-cell bold/colour Styler function
  f210__row_style                  – per-row bold/colour Styler function
  f211__epw_location_map           – PyDeck scatter map for EPW locations
"""

from __future__ import annotations

from typing import List

import pandas as pd
import pydeck as pdk
import streamlit as st


# ── Chart geometry constants ──────────────────────────────────────────────────
BOX_CHART_MARGINS = dict(l=20, r=20, t=10, b=10)
BAR_CHART_MARGINS = dict(l=20, r=20, t=10, b=10)

BAR_CHART_HEIGHT_ITEM_THRESHOLD = 30
BAR_CHART_LINE_HEIGHT_PX        = 15
BAR_CHART_HEIGHT_PADDING_PX     = 0
BAR_CHART_MAX_ROWS              = 50
BAR_CHART_HEIGHT_MAX_PX         = BAR_CHART_MAX_ROWS * BAR_CHART_LINE_HEIGHT_PX
BAR_CHART_HEIGHT_MIN_PX         = 300

# ── Catalogue typography ──────────────────────────────────────────────────────
CATALOGUE_LINE_HEIGHT  = 1.35
CATALOGUE_FONT_SIZE_PX = 13

# ── Colour maps ───────────────────────────────────────────────────────────────
CLIMATE_COLOR_MAP = {
    "Solar Irradiance": "#F7931E",
    "Temperature":      "#e63946",
    "Wind Speed":       "#457b9d",
    "AC Energy":        "#2a7d2e",
    "Monthly":          "#00a0e6",
}

# ── PV metric card CSS ────────────────────────────────────────────────────────
_PV_CARD_CSS = """
.pv-metric-card {
    background: #f8fdf8; border: 1px solid #c8e6c9;
    border-radius: 8px; padding: 0.75rem 0.9rem; margin-bottom: 0.2rem;
}
.pv-metric-card .label {
    font-size: 0.85rem; color: #555; text-transform: uppercase;
    letter-spacing: 0.02em; margin-bottom: 2px;
}
.pv-metric-card .value {
    font-family: 'Source Serif Pro', Georgia, serif !important;
    font-weight: 600; font-size: 1.6rem; color: #2a7d2e;
}
.pv-metric-card .unit { font-size: 1rem; color: #555; margin-left: 4px; }
.pv-metric-card--primary { line-height: 1.1; padding: 0.5rem 1.0rem; }
.pv-metric-card--primary .label { font-size: 0.88rem; margin-bottom: 0; }
.pv-metric-card--primary .value { font-size: 2rem; }
.pv-metric-card--primary .note  { font-size: 0.85rem; margin-top: 2px; line-height: 1.2; }
.pv-metric-card--leed { border: 3px solid #2a7d2e !important; text-align: center; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# f203
# ─────────────────────────────────────────────────────────────────────────────
def f203__inject_global_css() -> None:
    """
    Inject all global CSS (fonts, layout, typography, component styles).
    Called once per page from f202__create_page_header().
    """
    font_links = (
        '<link href="https://fonts.googleapis.com/css2?'
        'family=Source+Serif+Pro:ital,wght@1,400;1,600&display=swap" rel="stylesheet">'
        '<link href="https://fonts.googleapis.com/css2?'
        'family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200"'
        ' rel="stylesheet">'
        '<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">'
    )
    st.markdown(font_links, unsafe_allow_html=True)

    css = """
    .block-container {
        padding-top: 0.25rem; padding-bottom: 0.5rem;
        padding-left: 2rem;  padding-right: 2rem;
    }
    section[data-testid=stSidebar] .block-container { padding-top: 0.25rem; }
    div[data-testid=stAppViewContainer] > .main .block-container { padding-top: 0.25rem; }
    .main { padding-left: 1rem; padding-right: 1rem; }

    h2.custom-title, .custom-title {
        font-family: 'Source Serif Pro', Georgia, serif !important;
        font-style: italic; font-weight: 400; color: #2a7d2e;
        margin: 0; padding: 0;
    }
    .custom-caption { font-size: 0.8rem; color: gray; margin: 0; padding: 0; line-height: 1.3; }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        font-family: 'Source Serif Pro', Georgia, serif !important;
        font-style: italic; font-weight: 400;
    }

    .co2-value {
        font-family: 'Source Serif Pro', Georgia, serif !important;
        font-style: italic; font-weight: 600;
        font-size: 16px; color: #2a7d2e; margin: 0 0 8px 0; padding: 0;
    }
    .co2-value span {
        font-family: 'Source Serif Pro', Georgia, serif !important;
        font-style: italic; font-weight: 600;
        font-size: 16px; margin: 0; padding: 1px 3px;
    }

    .material-symbols-rounded, .material-symbols-outlined,
    [class*=material-symbols] {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important; font-style: normal !important;
        letter-spacing: normal !important; text-transform: none !important;
        display: inline-block; white-space: nowrap;
        -webkit-font-smoothing: antialiased;
    }

    .product-title   { font-size: 16px; font-weight: 600; }
    .product-details { font-size: %(px)spx; line-height: %(lh)s; }
    .brand           { font-size: %(px)spx; line-height: %(lh)s; color: #666; }

    [data-testid=stImage], .stImage { width: 100%% !important; }
    .stImage img {
        width: 100%% !important; height: auto !important;
        object-fit: contain !important;
        border: 1px solid #e8e8e8 !important; border-radius: 4px !important;
    }

    .hr-line {
        margin-top: -5px; margin-bottom: -10px;
        padding: 0; height: 1px; border: none; background-color: #ccc;
    }
    .hr-line-compact { margin-top: -5px; margin-bottom: -5px; }
    """ % {"px": CATALOGUE_FONT_SIZE_PX, "lh": CATALOGUE_LINE_HEIGHT}

    full_html = f"<style>{css}{_PV_CARD_CSS}</style>"
    try:
        st.html(full_html)
    except Exception:
        st.markdown(full_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# f204
# ─────────────────────────────────────────────────────────────────────────────
def f204__custom_divider(
    spacing_above_px: int = 0,
    spacing_below_px: int = 0,
    line_width_pct: int = 100,
    line_color: str = "#ccc",
    line_thickness_px: int = 1,
) -> None:
    """
    Render a horizontal divider with configurable spacing, width, colour,
    and thickness.
    """
    st.markdown(
        f"""
        <div style="margin-top: {spacing_above_px}px; margin-bottom: {spacing_below_px}px;">
            <hr style="width: {line_width_pct}%; height: {line_thickness_px}px;
                background-color: {line_color}; border: none; margin: 0; padding: 0;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# f205
# ─────────────────────────────────────────────────────────────────────────────
def f205__custom_hr() -> None:
    """Legacy compact horizontal rule (uses the .hr-line CSS class)."""
    st.markdown("""
        <style>
            .hr-line {
                margin-top: -5px; margin-bottom: -10px;
                padding: 0px; height: 1px; border: none;
                background-color: #ccc;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="hr-line"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# f206
# ─────────────────────────────────────────────────────────────────────────────
def f206__tight_hr_spacing() -> None:
    """Inject CSS that tightens vertical spacing around block-containers."""
    st.markdown("""
        <style>
            .block-container { padding-top: 0rem; padding-bottom: 0rem; }
            .stMarkdown p    { margin-top: -20px; margin-bottom: -20px; }
            .stButton > button { margin-top: 0px; margin-bottom: 0px; }
        </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# f207
# ─────────────────────────────────────────────────────────────────────────────
def f207__custom_filled_region(
    custom_text: str,
    bg_color: str | None = None,
    text_color: str = "black",
    border_style: str = "bottom",
    border_color: str = "black",
    border_thickness: str = "2px",
    width_percentage: int = 80,
) -> None:
    """
    Render a chart/section title using markdown ##### style.
    Styling params (bg_color, etc.) are ignored.
    """
    st.markdown(f"##### {custom_text}")


# ─────────────────────────────────────────────────────────────────────────────
# f208
# ─────────────────────────────────────────────────────────────────────────────
def f208__style_df(
    df: pd.DataFrame,
    range_min_max,
    colormap: str,
) -> "pd.io.formats.style.Styler":
    """
    Return a gradient-background-coloured Pandas Styler for *df*.

    Parameters
    ----------
    range_min_max : tuple[float, float] | None
        Explicit (min, max) for the colour scale, or ``None`` for data range.
    colormap : str
        Any matplotlib-compatible colormap name.
    """
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    if not range_min_max:
        range_min, range_max = numeric_df.min().min(), numeric_df.max().max()
    else:
        range_min, range_max = range_min_max

    return numeric_df.style.background_gradient(
        cmap=colormap, axis=None, vmin=range_min, vmax=range_max,
    ).format(precision=1)


# ─────────────────────────────────────────────────────────────────────────────
# f209
# ─────────────────────────────────────────────────────────────────────────────
def f209__highlight_odd_even_cells(val) -> str:
    """Pandas Styler function: bold + blue for odd numbers, green for even."""
    if isinstance(val, (int, float)):
        return "font-weight: bold; color: blue;" if val % 2 != 0 else "color: green;"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# f210
# ─────────────────────────────────────────────────────────────────────────────
def f210__row_style(df: pd.DataFrame, row_index: int) -> List[str]:
    """
    Pandas Styler row function: bold for rows 0 and 2, teal colour otherwise.
    """
    if row_index in [0, 2]:
        return ["font-weight: bold;"] * len(df.columns)
    return ["color: #65b3aa;"] * len(df.columns)


# ─────────────────────────────────────────────────────────────────────────────
# f211
# ─────────────────────────────────────────────────────────────────────────────
def f211__epw_location_map(
    data,
    col_lat: str,
    col_lon: str,
    zoom_level: int = 10,
    chart_height: int = 500,
    chart_width: int = 700,
    dot_size: int = 200,
    dot_color: list = None,
    dot_opacity: int = 160,
) -> None:
    """
    Render a PyDeck scatter map showing EPW file locations.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain *col_lat* and *col_lon* columns.
    dot_color : list[int]
        RGB list, default ``[255, 0, 0]``.
    """
    if dot_color is None:
        dot_color = [255, 0, 0]

    layer = pdk.Layer(
        "ScatterplotLayer",
        data,
        get_position=f"[{col_lon}, {col_lat}]",
        get_radius=dot_size,
        get_fill_color=dot_color + [dot_opacity],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=data[col_lat].mean(),
        longitude=data[col_lon].mean(),
        zoom=zoom_level,
        pitch=0,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v9",
        ),
        use_container_width=False,
        height=chart_height,
    )


# ── Backward-compatible aliases ───────────────────────────────────────────────
inject_global_css        = f203__inject_global_css
custom_divider           = f204__custom_divider
custom_hr                = f205__custom_hr
absrd_tight_hr_spacing   = f206__tight_hr_spacing
custom_filled_region     = f207__custom_filled_region
absrd_style_df_streamlit = f208__style_df
row_style                = f210__row_style
absrd__epw_location_map  = f211__epw_location_map
