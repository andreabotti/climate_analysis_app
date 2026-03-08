# IMPORT LIBRARIES
import os, requests, pathlib, json
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ladybug.epw import EPW

from libs.fn__data import f116__fetch_pv_production_data as fetch_pv_production_data, f121__iterate_pv_production as iterate_pv_production
from libs.fn__data import *
from libs.fn__charts import *


mapbox_access_token = 'pk.eyJ1IjoiYW5kcmVhYm90dGkiLCJhIjoiY2xuNDdybms2MHBvMjJqbm95aDdlZ2owcyJ9.-fs8J1enU5kC3L4mAJ5ToQ'



# ── Shared utilities ───────────────────────────────────────────────────────
from libs.fn__header import create_page_header
from libs.fn__ui import CLIMATE_COLOR_MAP, BOX_CHART_MARGINS, f204__custom_divider as custom_divider



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


# LOAD DATA INTO SESSION STATE
epw_files_list = st.session_state['epw_files_list']
epw_names = [item['epw_file_name'] for item in epw_files_list]
epw_files = [item['epw_object'] for item in epw_files_list]
st.session_state['epw_names'] = epw_names
st.session_state['epw_files'] = epw_files



####################################################################################
# LOCATION — EPW files if available, manual lat/lon fallback otherwise
# Builds a unified `locations` list used by the fetch loop below.
# Each entry: {'label': str, 'lat': float, 'lon': float, 'epw_name': str|None, 'epw_file': obj|None}

epw_locations = [
    {
        'label':    item['epw_file_name'],
        'lat':      item['epw_object'].location.latitude,
        'lon':      item['epw_object'].location.longitude,
        'epw_name': item['epw_file_name'],
        'epw_file': item['epw_object'],
    }
    for item in epw_files_list
]



####################################################################################


with st.sidebar:
    # st.markdown("### 🏆 PVWatts Inputs")

    # ── Location ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("#### 📍 Location")

    if epw_locations:
        # Use all loaded EPW files — no selectbox needed, loop covers them all
        st.sidebar.info(f"{len(epw_locations)} EPW file(s) loaded — all locations will be queried.", icon="📂")
        locations = epw_locations
    else:
        st.sidebar.info("No EPW files loaded — enter coordinates manually.", icon="📂")
        _sc1, _sc2 = st.sidebar.columns(2)
        _lat = _sc1.number_input("Latitude (°N)",  value=44.4949, min_value=-90.0,  max_value=90.0,  format="%.3f")
        _lon = _sc2.number_input("Longitude (°E)", value=11.3426, min_value=-180.0, max_value=180.0, format="%.3f")
        locations = [{'label': f"Manual ({_lat:.4f}, {_lon:.4f})", 'lat': _lat, 'lon': _lon, 'epw_name': 'manual', 'epw_file': None}]





    # ── System sizing ─────────────────────────────────────────────────────

        custom_divider(spacing_above_px=15, spacing_below_px=5)

    st.markdown("#### ⚡ System Sizing")


    # Use most recent year for which PVGIS data is available (PVGIS has no future-year data)
    _recent_year = min(datetime.now().year - 1, 2024)
    start_year = end_year = max(_recent_year, 2005)

    col_sidebar_1, col_sidebar_2 = st.sidebar.columns([2, 2])

    peak_power = col_sidebar_1.number_input(
        label="Peak Power (kWp)", min_value=0.1, max_value=10.0, value=1.0, step=0.1
    )
    system_loss = col_sidebar_2.slider(
        label="System Loss (%)", min_value=0, max_value=30, value=14, step=1
    )

    # ── Geometry & losses ─────────────────────────────────────────────────

    custom_divider(spacing_above_px=15, spacing_below_px=5)

    st.markdown("#### 📐 Geometry")

    sub_col_1, sub_col_2 = st.columns([1, 1], gap="small")
    with sub_col_1:
        tilt    = st.slider("Tilt (°)",          min_value=0,   max_value=90,  value=25)
    with sub_col_2:
        azimuth = st.slider("Azimuth (°)",       min_value=0,   max_value=359, value=180,
                        help="180° = south-facing (northern hemisphere).")





    custom_divider(spacing_above_px=15, spacing_below_px=30)

    fetch_btn = st.sidebar.button("▶  Fetch PVGIS Data", type="primary", width='stretch')

    # ── Tilt × Azimuth matrix — checkbox + button (replaces standalone button) ──
    run_matrix = st.sidebar.checkbox(
        "Run tilt × azimuth sensitivity analysis",
        value=False,
        help="Iterates over a grid of tilt and azimuth values. Calculation time > 2 minutes.",
    )
    if run_matrix:
        matrix_btn = st.sidebar.button("▶  Run PVGIS Matrix", type="secondary", width='stretch')
    else:
        matrix_btn = False







####################################################################################
# Helper: SD band chart
####################################################################################
MONTHS_SHORT = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
GREEN       = '#2a7d2e'
ORANGE      = '#F7931E'
BAND_ALPHA  = 0.15


def _plot_monthly_sd_chart(df_monthly: pd.DataFrame, city: str) -> go.Figure:
    """
    For each EPW file's monthly PVGIS output, draw:
      • Thick horizontal tick marks at the mean monthly energy (E_m)
      • Low-opacity ±SD bands (SD_m)
      • A secondary axis overlay for monthly irradiance (H(i)_m)
    Returns a Plotly Figure.
    """
    fig = go.Figure()

    # ── Identify column names defensively ────────────────────────────────
    cols = df_monthly.columns.tolist()

    # Energy columns
    em_col  = next((c for c in cols if c in ('E_m',  'monthly_energy', 'E_month')), None)
    sd_col  = next((c for c in cols if c in ('SD_m', 'SD_month', 'sd_m')),          None)
    him_col = next((c for c in cols if 'H(i)' in c and '_m' in c or c == 'monthly_irr'), None)

    if em_col is None:
        fig.add_annotation(text="Column E_m not found in PVGIS output", showarrow=False)
        return fig

    # Align to 12 months
    n = min(len(df_monthly), 12)
    months = MONTHS_SHORT[:n]
    x_idx  = list(range(n))   # numeric x so shapes align with traces
    em_vals  = df_monthly[em_col].values[:n].astype(float)
    sd_vals  = df_monthly[sd_col].values[:n].astype(float) if sd_col else np.zeros(n)
    him_vals = df_monthly[him_col].values[:n].astype(float) if him_col else None

    # ── ±SD bands (one shape per month) ──────────────────────────────────
    for i, (val, sd) in enumerate(zip(em_vals, sd_vals)):
        fig.add_shape(
            type="rect",
            x0=i - 0.35, x1=i + 0.35,
            y0=val - sd,  y1=val + sd,
            fillcolor=f"rgba(42,125,46,{BAND_ALPHA})",
            line=dict(width=0),
            layer="below",
        )

    # ── Thick horizontal mean lines ───────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_idx,
        y=em_vals,
        mode='markers',
        name='Monthly Energy (kWh)',
        marker=dict(
            symbol='line-ew-open',
            size=32,
            line=dict(width=5, color=GREEN),
        ),
        hovertemplate='<b>%{text}</b><br>Energy: %{y:,.1f} kWh<extra></extra>',
        text=months,
    ))

    # ── SD whiskers as invisible scatter for legend entry ────────────────
    if sd_col and sd_vals.any():
        fig.add_trace(go.Scatter(
            x=x_idx,
            y=em_vals,
            error_y=dict(
                type='data',
                array=sd_vals.tolist(),
                visible=True,
                color=f"rgba(42,125,46,0.4)",
                thickness=1.5,
                width=0,
            ),
            mode='markers',
            marker=dict(size=0, color='rgba(0,0,0,0)'),
            name='± SD',
            showlegend=True,
            hoverinfo='skip',
        ))

    # ── Irradiance overlay (secondary y-axis) ────────────────────────────
    if him_vals is not None:
        fig.add_trace(go.Scatter(
            x=x_idx,
            y=him_vals,
            mode='lines+markers',
            name='POA Irradiance (kWh/m²)',
            yaxis='y2',
            line=dict(color=ORANGE, width=2, dash='dot'),
            marker=dict(size=5, color=ORANGE),
            hovertemplate='<b>%{text}</b><br>Irradiance: %{y:,.1f} kWh/m²<extra></extra>',
            text=months,
        ))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=x_idx,
            ticktext=months,
            showgrid=False,
        ),
        yaxis=dict(title='Monthly Energy (kWh)', showgrid=True, gridcolor='#f0f0f0'),
        yaxis2=dict(
            title='POA Irradiance (kWh/m²)',
            overlaying='y', side='right', showgrid=False,
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family='Source Serif Pro, Georgia, serif', size=12),
        title=dict(text=city, font=dict(size=13, color='#555'), x=0.01),
    )
    return fig


####################################################################################
# Fetch PVGIS Data
####################################################################################
if fetch_btn:
    with st.spinner('Fetching data from PVGIS API...'):
        hourly_data_dict       = {}
        pv_production_data_dict = {}
        df_hourly_dict         = {}
        df_pv_monthly_dict     = {}
        df_pv_totals_dict      = {}
        bullet_points_dict     = {}

        for loc in locations:
            epw_name = loc['epw_name']
            _lat     = loc['lat']
            _lon     = loc['lon']

            hourly_data = fetch_pvgis_hourly_data(
                lat=_lat, lon=_lon,
                startyear=start_year, endyear=end_year,
                peakpower=peak_power, loss=system_loss,
                angle=tilt, azimuth=azimuth,
            )
            pv_production_data = fetch_pv_production_data(
                lat=_lat, lon=_lon,
                peakpower=peak_power, loss=system_loss,
                tilt=tilt, azimuth=azimuth,
            )

            hourly_data_dict[epw_name]        = hourly_data
            pv_production_data_dict[epw_name] = pv_production_data

            outputs   = hourly_data.get('outputs', {}) if hourly_data else {}
            df_hourly = convert_outputs_to_dataframe(outputs)
            df_hourly_dict[epw_name] = df_hourly

            if pv_production_data:
                df_pv_monthly, df_pv_totals = parse_pv_production_data(pv_production_data)
                bullet_points = extract_meta_monthly_variable_info(pv_production_data)
            else:
                df_pv_monthly, df_pv_totals, bullet_points = pd.DataFrame(), pd.DataFrame(), []

            df_pv_monthly_dict[epw_name]  = df_pv_monthly
            df_pv_totals_dict[epw_name]   = df_pv_totals
            bullet_points_dict[epw_name]  = bullet_points

        # Cache in session state so results survive reruns
        st.session_state['pvgis_hourly_data_dict']        = hourly_data_dict
        st.session_state['pvgis_pv_production_data_dict'] = pv_production_data_dict
        st.session_state['pvgis_df_hourly_dict']          = df_hourly_dict
        st.session_state['pvgis_df_pv_monthly_dict']      = df_pv_monthly_dict
        st.session_state['pvgis_df_pv_totals_dict']       = df_pv_totals_dict
        st.session_state['pvgis_bullet_points_dict']      = bullet_points_dict
        st.session_state['pvgis_params'] = dict(
            tilt=tilt, azimuth=azimuth,
            peak_power=peak_power, system_loss=system_loss,
        )


####################################################################################
# Display results (from session state — persists across reruns)
####################################################################################
if 'pvgis_df_pv_monthly_dict' in st.session_state:

    hourly_data_dict        = st.session_state['pvgis_hourly_data_dict']
    pv_production_data_dict = st.session_state['pvgis_pv_production_data_dict']
    df_hourly_dict          = st.session_state['pvgis_df_hourly_dict']
    df_pv_monthly_dict      = st.session_state['pvgis_df_pv_monthly_dict']
    df_pv_totals_dict       = st.session_state['pvgis_df_pv_totals_dict']
    bullet_points_dict      = st.session_state['pvgis_bullet_points_dict']
    _p                      = st.session_state['pvgis_params']

    # Invalidate cache if EPW list changed (e.g. user added/removed files after fetch)
    current_epw_names = {loc['epw_name'] for loc in locations}
    cached_keys = set(df_pv_monthly_dict.keys())
    if not current_epw_names.issubset(cached_keys):
        for k in ['pvgis_hourly_data_dict', 'pvgis_pv_production_data_dict', 'pvgis_df_hourly_dict',
                  'pvgis_df_pv_monthly_dict', 'pvgis_df_pv_totals_dict', 'pvgis_bullet_points_dict', 'pvgis_params']:
            st.session_state.pop(k, None)
        st.session_state['pvgis_cache_invalidated'] = True
        st.rerun()

    tabs = st.tabs(["📊 Charts", "📈 Monthly + SD", "📋 Tables", "📄 Raw Output"])

    # ── Tab 1: original subplots chart ───────────────────────────────────
    with tabs[0]:
        st.markdown('##### PVGIS Data Visualization — Charts')
        chart_cols = st.columns(len(locations))
        for i, (col, loc) in enumerate(zip(chart_cols, locations)):
            epw_name = loc['epw_name']
            epw_file = loc['epw_file']
            with col:
                _lat = round(loc['lat'], 3)
                _lon = round(loc['lon'], 3)
                station = epw_file._location.city if epw_file else loc['label']
                st.markdown(
                    f"<h5 style='text-align:center;color:black;'>{station} &nbsp; lat:{_lat} lon:{_lon}</h5>",
                    unsafe_allow_html=True,
                )
                fig_plotly = plot_pv_monthly_comparison_subplots(
                    df_pv_monthly_dict[epw_name], _p['tilt'], _p['azimuth'],
                    0, 265, 0, 265,
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                st.plotly_chart(fig_plotly, width='stretch', key=f"pvgis_chart_{i}")

    # ── Tab 2: Monthly energy + SD bands ─────────────────────────────────
    with tabs[1]:
        st.markdown('##### Monthly Energy with Standard Deviation (SD) bands')
        st.caption(
            "Thick horizontal lines = mean monthly energy (kWh). "
            "Shaded bands = ± SD across years. "
            "Dotted line = monthly POA irradiance (right axis)."
        )
        chart_cols = st.columns(len(locations))
        for i, (col, loc) in enumerate(zip(chart_cols, locations)):
            epw_name = loc['epw_name']
            epw_file = loc['epw_file']
            with col:
                df_monthly = df_pv_monthly_dict[epw_name]
                if not df_monthly.empty:
                    city = epw_file._location.city if epw_file else loc['label']
                    fig_sd = _plot_monthly_sd_chart(df_monthly, city)
                    st.plotly_chart(fig_sd, width='stretch', key=f"pvgis_sd_{i}")
                else:
                    st.info("No monthly data available for this location.")

    # ── Tab 3: Tables ─────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown('##### PVGIS Data Tables')
        chart_cols = st.columns(len(locations), gap="large")
        for col, loc in zip(chart_cols, locations):
            epw_name = loc['epw_name']
            epw_file = loc['epw_file']
            city = epw_file._location.city if epw_file else loc['label']
            with col:
                df_monthly = df_pv_monthly_dict[epw_name]
                if not df_monthly.empty:
                    df_daily, df_monthly_vars = split_and_transpose_pv_monthly(df_monthly)

                    if not df_daily.empty:
                        st.write(f"**Daily averages** ({city})")
                        st.dataframe(df_daily.round(2), width='stretch', height=120)
                        for bullet in extract_meta_variable_info_by_type(
                            pv_production_data_dict.get(epw_name) or {}, daily_only=True
                        ):
                            st.caption(bullet)

                    custom_divider(spacing_above_px=15, spacing_below_px=5)


                    if not df_monthly_vars.empty:
                        st.write(f"**Monthly data** ({city})")
                        st.dataframe(df_monthly_vars.round(2), width='stretch', height=180)
                        for bullet in extract_meta_variable_info_by_type(
                            pv_production_data_dict.get(epw_name) or {}, daily_only=False
                        ):
                            st.caption(bullet)

                if not df_pv_totals_dict[epw_name].empty:
                    custom_divider(spacing_above_px=15, spacing_below_px=5)

                    st.write(f"**Yearly Totals** ({city})")
                    st.dataframe(df_pv_totals_dict[epw_name])

    # ── Tab 4: Raw JSON ───────────────────────────────────────────────────
    with tabs[3]:
        st.markdown('##### PVGIS Data — Raw Output')
        if all(hourly_data_dict.get(k) is None for k in hourly_data_dict):
            st.caption("Hourly time series (seriescalc) is currently unavailable. Monthly data above comes from PVcalc.")
        chart_cols = st.columns(len(locations))
        for col, loc in zip(chart_cols, locations):
            epw_name = loc['epw_name']
            with col:
                with st.expander(f"Hourly Data (Raw JSON) — {epw_name}"):
                    st.json(hourly_data_dict[epw_name])
                with st.expander(f"PV Production Data (Raw JSON) — {epw_name}"):
                    st.json(pv_production_data_dict[epw_name])

else:
    # ── Empty state ───────────────────────────────────────────────────────
    if st.session_state.pop('pvgis_cache_invalidated', False):
        st.warning("EPW list changed — PVGIS cache cleared. Click **Fetch PVGIS Data** to reload.")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 0;color:#888;">
                <div style="font-size:3rem;">☀️</div>
                <div style="font-family:'Source Serif Pro',Georgia,serif;font-style:italic;
                            font-size:1.4rem;color:#2a7d2e;margin:0.5rem 0;">
                    Ready for PVGIS analysis
                </div>
                <div style="font-size:0.9rem;">
                    Configure system parameters in the sidebar<br/>
                    and click <strong>Fetch PVGIS Data</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


####################################################################################
# Tilt × Azimuth sensitivity matrix
####################################################################################
azimuth_values = range(-180, 181, 30)
tilt_values    = range(0, 91, 10)

if matrix_btn:
    with st.spinner('Fetching PVGIS data for tilt × azimuth matrix (this may take > 2 minutes)…'):

        chart_cols = st.columns(len(locations))
        for col, loc in zip(chart_cols, locations):
            epw_file = loc['epw_file']
            with col:
                _lat    = round(loc['lat'], 3)
                _lon    = round(loc['lon'], 3)
                station = epw_file._location.city if epw_file else loc['label']
                st.markdown(
                    f"<h5 style='text-align:center;color:black;'>{station} &nbsp; lat:{_lat} lon:{_lon}</h5>",
                    unsafe_allow_html=True,
                )

                data = iterate_pv_production(
                    lat=loc['lat'], lon=loc['lon'],
                    peakpower=peak_power, loss=system_loss,
                    azimuth_values=azimuth_values, tilt_values=tilt_values,
                    pause_duration=0.2,
                )

                if data:
                    df = pd.DataFrame(data)

                    output_cols = df.columns.drop(['azimuth', 'tilt', 'daily_energy', 'daily_irr', 'total_loss'])

                    metrics_dict = {
                        'monthly_energy': {'metric': '[kWh/mo]',    'range': [0, 140],  'colormap': 'BuGn'},
                        'yearly_energy':  {'metric': '[kWh/yr]',    'range': [0, 1500], 'colormap': 'BuGn'},
                        'monthly_irr':    {'metric': '[kWh/m²/mo]', 'range': [0, 200],  'colormap': 'Oranges'},
                        'yearly_irr':     {'metric': '[kWh/m²/yr]', 'range': [0, 2000], 'colormap': 'Oranges'},
                    }

                    for oc in output_cols:
                        df_filtered   = df[['azimuth', 'tilt', oc]]
                        energy_matrix = df_filtered.pivot(index='tilt', columns='azimuth', values=oc).round(0)
                        details       = metrics_dict.get(oc, {})
                        var_metric    = details.get('metric', '')
                        rng           = details.get('range', [0, 1500])
                        cmap          = details.get('colormap', 'BuGn')

                        st.write('')
                        st.markdown(f'##### {oc} — {var_metric}')
                        energy_matrix = energy_matrix.applymap(lambda x: f"{x:.0f}")
                        styled = absrd_style_df_streamlit(df=energy_matrix, range_min_max=rng, colormap=cmap)
                        st.dataframe(styled, width='stretch')

                    custom_divider(spacing_above_px=15, spacing_below_px=5)

                    with st.expander('Raw JSON data'):
                        st.write(data)
