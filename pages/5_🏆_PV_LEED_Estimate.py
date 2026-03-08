"""
pages/5_🏆_PV_LEED_Estimate.py
================================
PV Yield Estimation using NREL PVWatts V8 — LEED compliance focus.

Automatically uses lat/lon from EPW files already loaded in session state.
Falls back to manual coordinate entry if no EPW files are uploaded.

LEED credits addressed:
  • EA Credit: Renewable Energy Production (v4 / v4.1)

Key outputs:
  • Annual & monthly AC energy
  • Specific yield (kWh/kWp)
  • Performance ratio
  • Renewable energy fraction vs building total
  • LEED point estimation table
  • CSV export (monthly + optional hourly)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st



# PAGE HEADER
from libs.fn__header import create_page_header

# Fixed size logo with margins
create_page_header(
    logo_width="2500px",
    logo_height="auto",
    logo_max_height="70px",
    logo_margin="65px 10px 0 0",
    logo_padding="0px",
)



# Sidebar 10% wider for lat/lon widgets side by side
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { width: 20rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)




# ── LEED PV metric margin (customise in code) ───────────────────────────────
PV_METRIC_MARGIN_TOP_PX = 0
PV_METRIC_MARGIN_BOTTOM_PX = 40
PV_METRIC_MARGIN_LEFT_PX = 0
PV_METRIC_MARGIN_RIGHT_PX = 0

# ── Shared utilities ───────────────────────────────────────────────────────
from libs.fn__header import create_page_header
from libs.fn__ui import CLIMATE_COLOR_MAP, BOX_CHART_MARGINS, f204__custom_divider as custom_divider
from libs.fn__data import (
    pvwatts_query,
    get_api_key,
    MONTHS,
    ARRAY_TYPE_LABELS,
    MODULE_TYPE_LABELS,
    PVWattsResult,
)

# create_page_header(
#     subtitle="PV Yield — LEED Compliance (PVWatts V8)",
#     caption=(
#         "NREL PVWatts V8-based PV production estimates for "
#         "<a href='https://www.usgbc.org/credits/new-construction/v4/energy-%26-atmosphere' "
#         "target='_blank'>LEED EA Renewable Energy Credit</a>."
#     ),
# )

# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE — pull EPW locations if available
# ══════════════════════════════════════════════════════════════════════════
epw_files_list: list = st.session_state.get("epw_files_list", [])
epw_locations: list[dict] = []
for item in epw_files_list:
    epw = item["epw_object"]
    loc = epw.location
    epw_locations.append({
        "label":  f"{loc.city} ({item['epw_file_name']})",
        "lat":    loc.latitude,
        "lon":    loc.longitude,
        "city":   loc.city,
    })




# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # st.markdown("### 🏆 PVWatts Inputs")

    # ── API key (from secrets only — not shown in UI) ────────────────────
    api_key = get_api_key()
    if not api_key:
        st.warning("NREL API key not found. Set NREL_API_KEY in .streamlit/secrets.toml to run PVWatts.", icon="🔑")



    # ── Location ─────────────────────────────────────────────────────────
    st.markdown("#### 📍 Location")

    if epw_locations:
        location_labels = [loc["label"] for loc in epw_locations]
        selected_label  = st.selectbox("Use location from EPW file", options=location_labels)
        selected_loc    = next(l for l in epw_locations if l["label"] == selected_label)
        lat = selected_loc["lat"]
        lon = selected_loc["lon"]
        st.caption(f"lat {lat:.4f}  |  lon {lon:.4f}")
        if st.checkbox("Override coordinates manually"):
            # lat = st.number_input("Latitude (°N)",  value=float(lat),  min_value=-90.0,  max_value=90.0,  format="%.4f")
            # lon = st.number_input("Longitude (°E)", value=float(lon),  min_value=-180.0, max_value=180.0, format="%.4f")
            _sc1, _sc2 = st.sidebar.columns(2)
            lat = _sc1.number_input("Latitude (°N)",  value=44.4949, min_value=-90.0,  max_value=90.0,  format="%.3f")
            lon = _sc2.number_input("Longitude (°E)", value=11.3426, min_value=-180.0, max_value=180.0, format="%.3f")

    else:
        st.info("No EPW files loaded — enter coordinates manually.", icon="📂")
        _sc1, _sc2 = st.sidebar.columns(2)
        lat = _sc1.number_input("Latitude (°N)",  value=44.4949, min_value=-90.0,  max_value=90.0,  format="%.3f")
        lon = _sc2.number_input("Longitude (°E)", value=11.3426, min_value=-180.0, max_value=180.0, format="%.3f")


    dataset = st.selectbox(
        "Solar Dataset",
        options=["intl", "nsrdb", "tmy3", "tmy2"],
        index=0,
        help="Use 'intl' for locations outside the USA (e.g. Europe).",
    )






    # ── System sizing ─────────────────────────────────────────────────────
    custom_divider(spacing_above_px=15, spacing_below_px=5)
    
    st.markdown("#### ⚡ System Sizing")

    sub_col_1, sub_col_2 = st.columns([1, 1], gap="small")

    with sub_col_1:
        capacity_kw = st.number_input(
            "System Size (kWp)",
            value=1.0, min_value=0.1, max_value=500_000.0, step=1.0,
        )
    with sub_col_2:
        module_type = st.selectbox(
            "Module Type",
            options=list(MODULE_TYPE_LABELS.keys()),
            format_func=lambda k: MODULE_TYPE_LABELS[k],
            index=0,
        )

    array_type = st.selectbox(
        "Array / Mounting Type",
        options=list(ARRAY_TYPE_LABELS.keys()),
        format_func=lambda k: ARRAY_TYPE_LABELS[k],
        index=1,
    )

    losses  = st.slider("System Losses (%)", min_value=0, max_value=30, value=14, step=1,
                        help="Wiring, soiling, shading, inverter, mismatch, etc. NREL default = 14 %."    )






    # ── Geometry & losses ─────────────────────────────────────────────────

    custom_divider(spacing_above_px=15, spacing_below_px=5)

    st.markdown("#### 📐 Geometry")

    sub_col_1, sub_col_2 = st.columns([1, 1], gap="small")
    with sub_col_1:
        tilt    = st.slider("Tilt (°)",          min_value=0,   max_value=90,  value=25)
    with sub_col_2:
        azimuth = st.slider("Azimuth (°)",       min_value=0,   max_value=359, value=180,
                        help="180° = south-facing (northern hemisphere).")


    # custom_divider(spacing_above_px=12, spacing_below_px=2)

    # ── Hourly option ─────────────────────────────────────────────────────
    include_hourly = st.checkbox(
        "Include Hourly Data",
        value=False,
        help="Fetches 8 760 hourly values — slower API call (~5–10 s extra).",
    )
    timeframe = "hourly" if include_hourly else "monthly"

    run_btn = st.button(
        "▶  Run PVWatts",
        type="primary",
        width='stretch',
        disabled=not api_key,
    )

# ══════════════════════════════════════════════════════════════════════════
# HELPER: styled metric card
# ══════════════════════════════════════════════════════════════════════════
def _card(label: str, value: str, unit: str = "", note: str = "", primary: bool = False, tooltip: str = "") -> str:
    extra_class = " pv-metric-card--primary" if primary else ""
    note_style = "" if primary else ' style="font-size:0.9rem;color:#666;margin-top:-5px;"'
    note_html = f'<div class="note"{note_style}>{note}</div>' if note else ""
    tooltip_icon = (
        f'<span class="pv-help-icon" title="{tooltip}" style="cursor:help;opacity:0.6;margin-left:4px;font-size:0.85em;">ⓘ</span>'
        if tooltip else ""
    )
    return (
        f'<div class="pv-metric-card{extra_class}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}<span class="unit"> {unit}</span>{tooltip_icon}</div>'
        f'{note_html}</div>'
    )

# ══════════════════════════════════════════════════════════════════════════
# RESULT RENDERER
# ══════════════════════════════════════════════════════════════════════════
def _render(r: PVWattsResult) -> None:
    # Build tab list: LEED tab always, Monthly if data, Hourly if data
    tab_names = ["LEED EA Credit - Ren. Energy"]
    if r.ac_monthly_kwh:
        tab_names.append("Monthly Production")
    if r.ac_hourly_w:
        tab_names.append("Hourly Profile")

    tabs = st.tabs(tab_names)








    # ═══ Tab 1: LEED EA Credit - Ren. Energy ═══
    with tabs[0]:
        st.markdown("##### LEED EA Credit — Renewable Energy Production")

        # ── KPI row (margin from PV_METRIC_MARGIN_* constants) ─────────────
        st.markdown(
            f"""
            <style>
            .pv-metric-card {{
                margin: {PV_METRIC_MARGIN_TOP_PX}px {PV_METRIC_MARGIN_RIGHT_PX}px {PV_METRIC_MARGIN_BOTTOM_PX}px {PV_METRIC_MARGIN_LEFT_PX}px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )


        # Building Annual Energy Use widget + LEED Points in c5 (decoupled from Run PVWatts)
        c1, c2, c3, spacing, widget_col, c4, c5 = st.columns([1, 1, 1, 0.1, 1, 1, 1])

        # Keep building_kwh available for re_frac and the results table below
        building_kwh = st.session_state.get("pvwatts_building_kwh", 10000)

        with c5:
            # ── LEED points card FIRST (uses current building_kwh from session_state/default) ──
            re_frac_preview = (r.ac_annual_kwh / building_kwh * 100) if building_kwh > 0 else 0.0
            re_frac_preview = min(re_frac_preview, 100.0)

            leed_points_preview = 0
            if re_frac_preview >= 10: leed_points_preview = 5
            elif re_frac_preview >= 5: leed_points_preview = 3
            elif re_frac_preview >= 2: leed_points_preview = 2
            elif re_frac_preview >= 1: leed_points_preview = 1


            st.markdown(
                _card(
                    "Estimated LEED Points",
                    f"{leed_points_preview:.0f}",
                    "/5",
                    "EA Renewable Energy Production",
                    tooltip="LEED v4.1: Fraction ≥1% → 1 point | ≥2% → 2 points | ≥5% → 3 points | ≥10% → 5 points",
                ),
                unsafe_allow_html=True,
            )



        with widget_col:

            # ── Building input SECOND ─────────────────────────────────────────
            st.markdown("###### Building Energy Use")
            # custom_divider(spacing_above_px=50, spacing_below_px=10)
            building_kwh = st.number_input(
                "Annual Energy Use (kWh)",
                value=int(building_kwh),
                min_value=0,
                step=500,
                key="pvwatts_building_kwh",
                help="Enter total building energy use to compute LEED renewable fraction and points.",
            )

        # Now compute “official” re_frac from the (possibly updated) building_kwh
        re_frac = (r.ac_annual_kwh / building_kwh * 100) if building_kwh > 0 else 0.0
        re_frac = min(re_frac, 100.0)

        c1.markdown(_card("Annual AC Energy",  f"{r.ac_annual_kwh:,.0f}", "kWh/yr",  "Net production at meter", primary=False), unsafe_allow_html=True)
        c2.markdown(_card("Specific Yield",    f"{r.specific_yield_kwh_kwp:,.0f}", "kWh/kWp", "AC per kWp installed"), unsafe_allow_html=True)
        c3.markdown(_card("Performance Ratio", f"{r.performance_ratio*100:.1f}", "%", "Typical range 70–85 %"), unsafe_allow_html=True)
        if building_kwh > 0:
            c4.markdown(_card("LEED RE Fraction", f"{re_frac:.1f}", "%", f"of {building_kwh:,.0f} kWh building use"), unsafe_allow_html=True)
        else:
            c4.markdown(_card("LEED RE Fraction", "—", "", "Enter building energy use to the right"), unsafe_allow_html=True)

        # (Optional) remove the old second `with c5:` block entirely (it’s now redundant)





        # ── System Parameters | Results table (side by side) ──

        col_params, spacing1, spacing2, spacing3, col_results = st.columns([2, 1, 1, 0.1, 2])

        with col_params:
            st.markdown(
                f"""
##### System Parameters

| Parameter | Value |
|-----------|-------|
| Location | lat {r.lat:.4f}, lon {r.lon:.4f} |
| Solar Dataset | {r.dataset.upper()} |
| System Capacity | {r.system_capacity_kw:.1f} kWp |
| Module Type | {MODULE_TYPE_LABELS[r.module_type]} |
| Array / Mounting | {ARRAY_TYPE_LABELS[r.array_type]} |
| Tilt / Azimuth | {r.tilt}° / {r.azimuth}° |
| System Losses | {r.losses_pct:.1f} % |
                """
            )
        with col_results:
            st.markdown(
                f"""
##### Results

| Metric | Value |
|--------|-------|
| Annual AC Production | **{r.ac_annual_kwh:,.0f} kWh/yr** |
| Annual DC Production | {r.dc_annual_kwh:,.0f} kWh/yr |
| Specific Yield | **{r.specific_yield_kwh_kwp:,.0f} kWh/kWp** |
| Performance Ratio | **{r.performance_ratio*100:.1f} %** |
| Avg Daily Irradiance | {r.solrad_annual:.2f} kWh/m²/day |
{f"| Building Energy Use | {building_kwh:,.0f} kWh/yr |" if building_kwh > 0 else ""}
{f"| **Renewable Energy Fraction** | **{re_frac:.1f} %** |" if building_kwh > 0 else ""}
                """
            )






    # ═══ Tab 2: Monthly Production ───────────────────────────────────────
    if r.ac_monthly_kwh:
        with tabs[1]:
            st.markdown("### Monthly Production")
            df_mo = pd.DataFrame({
                "Month":              MONTHS,
                "AC Energy (kWh)":    r.ac_monthly_kwh,
                "DC Energy (kWh)":    r.dc_monthly_kwh if r.dc_monthly_kwh else [0]*12,
                "POA Irradiance (kWh/m²)": r.poa_monthly if r.poa_monthly else [0]*12,
                "Avg Daily Irr (kWh/m²/d)": r.solrad_monthly if r.solrad_monthly else [0]*12,
            })

            tab_chart, tab_table, tab_export = st.tabs(["📊 Chart", "📋 Table", "📥 Export"])

            with tab_chart:
                fig = go.Figure()
                fig.add_bar(
                    x=df_mo["Month"],
                    y=df_mo["AC Energy (kWh)"],
                    name="AC Energy",
                    marker_color=CLIMATE_COLOR_MAP["AC Energy"],
                    hovertemplate="%{x}: <b>%{y:,.0f} kWh</b><extra></extra>",
                )
                if r.poa_monthly:
                    fig.add_scatter(
                        x=df_mo["Month"],
                        y=df_mo["POA Irradiance (kWh/m²)"],
                        name="POA Irradiance",
                        mode="lines+markers",
                        yaxis="y2",
                        line=dict(color=CLIMATE_COLOR_MAP["Solar Irradiance"], width=2),
                        marker=dict(size=6),
                        hovertemplate="%{x}: <b>%{y:,.0f} kWh/m²</b><extra></extra>",
                    )
                fig.update_layout(
                    yaxis=dict(title="AC Energy (kWh)", showgrid=True),
                    yaxis2=dict(title="POA Irradiance (kWh/m²)", overlaying="y", side="right", showgrid=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=BOX_CHART_MARGINS,
                    height=380,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(family="Source Serif Pro, Georgia, serif", size=12),
                )
                st.plotly_chart(fig, width='stretch')

            with tab_table:
                st.dataframe(
                    df_mo.style.format({
                        "AC Energy (kWh)": "{:,.0f}",
                        "DC Energy (kWh)": "{:,.0f}",
                        "POA Irradiance (kWh/m²)": "{:,.0f}",
                        "Avg Daily Irr (kWh/m²/d)": "{:.2f}",
                    }),
                    width='stretch',
                    hide_index=True,
                )

            with tab_export:
                st.download_button(
                    "⬇ Download monthly CSV",
                    data=df_mo.to_csv(index=False).encode("utf-8"),
                    file_name="pvwatts_monthly.csv",
                    mime="text/csv",
                )

    # ═══ Tab 3: Hourly Profile ───────────────────────────────────────────
    if r.ac_hourly_w:
        with tabs[2]:
            st.markdown("### Hourly Profile (8 760 hours)")
            df_hr = pd.DataFrame({
                "Hour":             range(len(r.ac_hourly_w)),
                "AC Power (W)":     r.ac_hourly_w,
                "POA (W/m²)":       r.poa_hourly,
                "Amb. Temp (°C)":   r.tamb_hourly,
                "Wind Speed (m/s)": r.wspd_hourly,
            })

            ht1, ht2, ht3 = st.tabs(["⚡ AC Power", "🌡 Temperature & Wind", "📥 Export"])

            with ht1:
                fig_h = px.line(
                    df_hr, x="Hour", y="AC Power (W)",
                    color_discrete_sequence=[CLIMATE_COLOR_MAP["AC Energy"]],
                    labels={"Hour": "Hour of Year"},
                )
                fig_h.update_layout(margin=BOX_CHART_MARGINS, height=320, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_h, width='stretch')

            with ht2:
                fig_tw = go.Figure()
                fig_tw.add_scatter(x=df_hr["Hour"], y=df_hr["Amb. Temp (°C)"],
                                   name="Temp (°C)", mode="lines",
                                   line=dict(color=CLIMATE_COLOR_MAP["Temperature"], width=1))
                fig_tw.add_scatter(x=df_hr["Hour"], y=df_hr["Wind Speed (m/s)"],
                                   name="Wind (m/s)", mode="lines", yaxis="y2",
                                   line=dict(color=CLIMATE_COLOR_MAP["Wind Speed"], width=1))
                fig_tw.update_layout(
                    yaxis=dict(title="Temperature (°C)"),
                    yaxis2=dict(title="Wind Speed (m/s)", overlaying="y", side="right"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=BOX_CHART_MARGINS, height=320,
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig_tw, width='stretch')

            with ht3:
                st.download_button(
                    "⬇ Download hourly CSV",
                    data=df_hr.to_csv(index=False).encode("utf-8"),
                    file_name="pvwatts_hourly.csv",
                    mime="text/csv",
                )


# ══════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════
if run_btn:
    if not api_key:
        st.error("Please provide an NREL API key.")
    else:
        with st.spinner("Querying NREL PVWatts V8…"):
            try:
                result = pvwatts_query(
                    api_key=api_key,
                    lat=float(lat),
                    lon=float(lon),
                    system_capacity_kw=float(capacity_kw),
                    tilt=float(tilt),
                    azimuth=float(azimuth),
                    losses_pct=float(losses),
                    module_type=int(module_type),
                    array_type=int(array_type),
                    timeframe=timeframe,
                    dataset=dataset,
                )
                st.session_state["pvwatts_result"] = result
            except Exception as exc:
                st.error(f"PVWatts query failed: {exc}")
                st.stop()

# Render cached result (survives sidebar interaction without re-querying)
if "pvwatts_result" in st.session_state:
    _render(st.session_state["pvwatts_result"])
else:
    # Empty state
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 0;color:#888;">
                <div style="font-size:3rem;">🏆</div>
                <div style="font-family:'Source Serif Pro',Georgia,serif;font-style:italic;
                            font-size:1.4rem;color:#2a7d2e;margin:0.5rem 0;">
                    Ready for PVWatts analysis
                </div>
                <div style="font-size:0.9rem;">
                    Configure system parameters in the sidebar<br/>
                    and click <strong>Run PVWatts</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
