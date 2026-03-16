"""
libs/fn__charts.py
==================
Chart and visualisation layer.  All Plotly / Ladybug figure builders.

Number range: 301–324  (charts series)

Absorbs:
  - libs/fn__chart_libraries.py  (full file)
  - chart functions from libs/fn__libraries.py:
      absrd_daytime_nighttime_scatter, plot_pv_monthly_comparison_subplots

Colour utilities and analysis-period helpers imported from fn__data.py.

Function index
--------------
  f301__avg_daily_profile                    – monthly subplots: daily scatter + median line
  f302__plot_hourly_line_chart_with_slider   – hourly line chart with date-range slider
  f303__convert_rgba_to_plotly_colorscale    – RGBA list → Plotly colorscale
  f304__plot_heatmap_from_datetime_index     – date × hour heatmap
  f305__slice_data_by_month                  – slice LB data into month-range DataFrame
  f306__bin_timeseries_data                  – bin time-series into value buckets
  f307__normalize_data                       – normalise binned counts to percentages
  f308__create_stacked_bar_chart             – monthly stacked-bar from binned data
  f309__get_figure_config                    – Plotly toolbar config (SVG download)
  f310__get_fields                           – dict of EPW field name → field index
  f311__get_diurnal_average_chart_figure     – diurnal average from EPW
  f312__get_hourly_data_figure               – hourly heatmap from HourlyContinuousCollection
  f313__get_bar_chart_figure                 – monthly/daily bar chart from EPW fields
  f314__get_hourly_line_chart_figure         – hourly line chart from HourlyContinuousCollection
  f315__get_hourly_diurnal_average_chart_figure  – diurnal average from HCC
  f316__get_daily_chart_figure               – daily bar chart from HCC
  f317__get_sunpath_figure                   – sunpath from EPW location
  f318__get_degree_days_figure               – HDD/CDD monthly bar chart
  f319__get_windrose_figure                  – Ladybug windrose from EPW
  f320__get_psy_chart_figure                 – psychrometric chart from EPW
  f321__line_chart_daily_range               – daily min/max/avg range line chart
  f322__plot_windrose                        – custom Plotly polar windrose
  f323__daytime_nighttime_scatter            – scatter of daytime vs night-time values
  f324__plot_pv_monthly_comparison_subplots  – side-by-side E_m / H(i)_m bar charts
"""

from __future__ import annotations

import calendar
from math import ceil, floor
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW, EPWFields
from ladybug.hourlyplot import HourlyPlot
from ladybug.legend import LegendParameters
from ladybug.monthlychart import MonthlyChart
from ladybug.psychchart import PsychrometricChart
from ladybug.sunpath import Sunpath
from ladybug.windrose import WindRose
from ladybug_charts.utils import Strategy
from ladybug_comfort.chart.polygonpmv import PolygonPMV
from ladybug_comfort.degreetime import cooling_degree_time, heating_degree_time
from ladybug.datatype.temperaturetime import CoolingDegreeTime, HeatingDegreeTime

# Import shared utilities from data layer
try:
    from libs.fn__data import (
        colorsets,
        f101__apply_analysis_period,
        f102__slice_df_analysis_period,
        f103__get_colors,
        f104__rgb_to_hex,
    )
except ImportError:
    from fn__data import (
        colorsets,
        f101__apply_analysis_period,
        f102__slice_df_analysis_period,
        f103__get_colors,
        f104__rgb_to_hex,
    )

# ── Internal shortcuts ────────────────────────────────────────────────────────
_apply_ap   = f101__apply_analysis_period
_slice_df   = f102__slice_df_analysis_period
_get_colors = f103__get_colors
_rgb_hex    = f104__rgb_to_hex


# ─────────────────────────────────────────────────────────────────────────────
# f301
# ─────────────────────────────────────────────────────────────────────────────
def f301__avg_daily_profile(
    plot_data: HourlyContinuousCollection,
    plot_analysis_period,
    global_colorset: str,
) -> Figure:
    """Monthly subplots of hourly scatter with median daily profile overlay."""
    var_name = str(plot_data.header.data_type)
    var_unit = str(plot_data.header.unit)
    range_y  = [5 * floor(plot_data.min / 5), 5 * ceil(plot_data.max / 5)]

    df = pd.DataFrame(plot_data.values, columns=["value"])
    df["datetime"] = pd.date_range(start="2023-01-01 00:00", periods=len(df), freq="h")
    df["month"]    = df["datetime"].dt.month
    df["hour"]     = df["datetime"].dt.hour
    df = _slice_df(df, plot_analysis_period)

    if df.empty:
        st.warning("No data available for the selected period.")
        return go.Figure()

    var_month_ave = df.groupby(["month", "hour"])["value"].median().reset_index()
    var_color     = colorsets[global_colorset]
    norm          = (df["value"] - df["value"].min()) / (df["value"].max() - df["value"].min())
    df["color"]   = norm.apply(lambda x: _rgb_hex(var_color[int(x * (len(var_color) - 1))][:3]))

    unique_months = np.sort(df["month"].unique())
    if len(unique_months) == 0:
        st.warning("No data available for the selected period.")
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=len(unique_months), shared_yaxes=True,
        subplot_titles=[pd.to_datetime(str(m), format="%m").strftime("%b") for m in unique_months],
        horizontal_spacing=0.01,
    )

    for i, month in enumerate(unique_months):
        md  = df[df["month"] == month]
        ma  = var_month_ave[var_month_ave["month"] == month]
        fig.add_trace(go.Scatter(
            x=md["hour"], y=md["value"], mode="markers",
            marker=dict(color=md["color"], opacity=0.5, size=3),
            name=pd.to_datetime(str(month), format="%m").strftime("%b"), showlegend=False,
            customdata=md["month"],
            hovertemplate=f"<b>{var_name}: {{%y:.2f}} {var_unit}</b><br>Month: %{{customdata}}<br>Hour: %{{x}}:00<br>",
        ), row=1, col=i + 1)
        fig.add_trace(go.Scatter(
            x=ma["hour"], y=ma["value"], mode="lines",
            line_color="black", line_width=2.5, showlegend=False,
            hovertemplate=f"<b>{var_name}: {{%y:.2f}} {var_unit}</b><br>Hour: %{{x}}:00<br>",
        ), row=1, col=i + 1)
        fig.update_xaxes(range=[0, 24], row=1, col=i + 1)
        fig.update_yaxes(range=range_y, row=1, col=i + 1)

    fig.update_xaxes(ticktext=["6", "12", "18"], tickvals=[6, 12, 18], tickangle=0)
    fig.update_layout(template="plotly_white", dragmode=False, height=400,
                      margin=dict(l=20, r=20, t=80, b=20), title=f"{var_name} ({var_unit})")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# f302
# ─────────────────────────────────────────────────────────────────────────────
def f302__plot_hourly_line_chart_with_slider(
    df: pd.DataFrame, variable: str, global_colorset: str
) -> None:
    """Render an hourly line chart with an interactive date-range slider."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df["datetime"])

    var_color = colorsets[global_colorset]
    norm      = (df[variable] - df[variable].min()) / (df[variable].max() - df[variable].min())
    df["color"] = norm.apply(lambda x: _rgb_hex(var_color[int(x * (len(var_color) - 1))][:3]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[variable], mode="lines",
                             line=dict(color=df["color"].iloc[0], width=2), name=variable))
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"),
                      title=f"Hourly Line Chart for {variable}",
                      xaxis_title="Time", yaxis_title=variable,
                      template="plotly_white", height=500, margin=dict(l=20, r=20, t=80, b=20))

    start_date, end_date = st.slider("Select Date Range",
                                     value=(df.index.min(), df.index.max()),
                                     format="YYYY-MM-DD HH:mm")
    filtered = df[(df.index >= start_date) & (df.index <= end_date)]
    fig.update_traces(x=filtered.index, y=filtered[variable], selector=dict(name=variable))
    st.plotly_chart(fig)


# ─────────────────────────────────────────────────────────────────────────────
# f303
# ─────────────────────────────────────────────────────────────────────────────
def f303__convert_rgba_to_plotly_colorscale(rgba_colors) -> list:
    """Convert a list of ``(R, G, B, A)`` tuples to a Plotly colorscale."""
    n = len(rgba_colors)
    return [[i / (n - 1), f"rgb({r},{g},{b})"] for i, (r, g, b, _a) in enumerate(rgba_colors)]


# ─────────────────────────────────────────────────────────────────────────────
# f304
# ─────────────────────────────────────────────────────────────────────────────
def f304__plot_heatmap_from_datetime_index(
    data: pd.DataFrame, value_col: str,
    agg_func: str = "sum", axis_min=None, axis_max=None, custom_colorscale=None,
) -> Figure:
    """Create a date × hour heatmap from a DataFrame with a datetime index."""
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data["Date"] = data.index.date
    data["Hour"] = data.index.hour
    pivot = data.pivot_table(index="Hour", columns="Date", values=value_col, aggfunc=agg_func)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns.astype(str), y=pivot.index,
        zmin=axis_min, zmax=axis_max, colorscale=custom_colorscale,
        colorbar=dict(title=value_col),
    ))
    fig.update_layout(xaxis_title="Date", yaxis_title="Hour",
                      template="plotly_white", margin=dict(l=30, r=30, t=40, b=20))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# f305
# ─────────────────────────────────────────────────────────────────────────────
def f305__slice_data_by_month(
    plot_data: HourlyContinuousCollection, start_month: int, end_month: int
) -> pd.DataFrame:
    """Return a DataFrame of hourly values filtered to *start_month*–*end_month*."""
    df = pd.DataFrame(plot_data.values, columns=["value"])
    df["datetime"] = pd.date_range(start="2023-01-01 00:00", periods=len(df), freq="h")
    df["month"]    = df["datetime"].dt.month
    return df[(df["month"] >= start_month) & (df["month"] <= end_month)]


# ─────────────────────────────────────────────────────────────────────────────
# f306
# ─────────────────────────────────────────────────────────────────────────────
def f306__bin_timeseries_data(
    df: pd.DataFrame, min_val: float, max_val: float, step: float
) -> pd.DataFrame:
    """Bin the ``"value"`` column of *df* and count occurrences by month."""
    bins   = np.arange(min_val, max_val + step, step)
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    df = df.copy()
    df["binned"] = pd.cut(df["value"], bins=bins, labels=labels, right=False)
    return df.groupby(["month", "binned"]).size().reset_index(name="count")


# ─────────────────────────────────────────────────────────────────────────────
# f307
# ─────────────────────────────────────────────────────────────────────────────
def f307__normalize_data(binned_data: pd.DataFrame) -> pd.DataFrame:
    """Add a ``"percentage"`` column (count / monthly total × 100)."""
    totals      = binned_data.groupby("month")["count"].sum().reset_index(name="total")
    binned_data = binned_data.merge(totals, on="month")
    binned_data["percentage"] = (binned_data["count"] / binned_data["total"]) * 100
    return binned_data


# ─────────────────────────────────────────────────────────────────────────────
# f308
# ─────────────────────────────────────────────────────────────────────────────
def f308__create_stacked_bar_chart(
    binned_data: pd.DataFrame, color_map: str, normalize: bool
) -> Figure:
    """Monthly stacked-bar chart from binned time-series data."""
    binned_data = binned_data.copy()
    binned_data["binned"] = binned_data["binned"].astype(str)
    binned_data["month"]  = binned_data["month"].apply(lambda x: calendar.month_abbr[int(x)])

    value_field = "percentage" if normalize else "count"
    var_color   = colorsets[color_map]
    df          = binned_data.copy()
    df["bin_min"] = df["binned"].apply(lambda x: x.split("-")[0]).astype(int)
    norm = (df["bin_min"] - df["bin_min"].min()) / (df["bin_min"].max() - df["bin_min"].min())
    df["color"] = norm.apply(lambda x: _rgb_hex(var_color[int(x * (len(var_color) - 1))][:3]))

    fig = go.Figure()
    for bin_value in binned_data["binned"].unique():
        bin_data = df[df["binned"] == bin_value]
        fig.add_trace(go.Bar(x=bin_data["month"], y=bin_data[value_field],
                             name=bin_value, marker_color=bin_data["color"].tolist()))

    fig.update_layout(barmode="stack", xaxis_title="Month",
                      yaxis_title="Percentage of Hours (%)" if normalize else "Total Hours",
                      height=450, margin=dict(l=30, r=30, t=40, b=10))
    fig.update_xaxes(type="category")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# f309
# ─────────────────────────────────────────────────────────────────────────────
def f309__get_figure_config(title: str) -> dict:
    """Return a Plotly toolbar config that enables SVG download."""
    return {"toImageButtonOptions": {"format": "svg", "filename": title,
                                     "height": 350, "width": 700, "scale": 1}}


# ─────────────────────────────────────────────────────────────────────────────
# f310
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def f310__get_fields() -> dict:
    """Return a dict mapping EPW variable name → field index (fields 6–33)."""
    return {EPWFields._fields[i]["name"].name: i for i in range(6, 34)}


# ─────────────────────────────────────────────────────────────────────────────
# f311
# ─────────────────────────────────────────────────────────────────────────────
def f311__get_diurnal_average_chart_figure(
    epw: EPW, global_colorset: str, switch: bool = False
) -> Figure:
    """Return a diurnal average chart from the full EPW dataset."""
    return epw.diurnal_average_chart(show_title=True, colors=_get_colors(switch, global_colorset))


# ─────────────────────────────────────────────────────────────────────────────
# f312
# ─────────────────────────────────────────────────────────────────────────────
def f312__get_hourly_data_figure(
    data: HourlyContinuousCollection, global_colorset: str,
    conditional_statement: str, min: float, max: float,
    st_month: int, st_day: int, st_hour: int,
    end_month: int, end_day: int, end_hour: int,
) -> Figure:
    """Hourly heatmap from a HCC, optionally filtered by conditional statement."""
    lb_ap = AnalysisPeriod(st_month, st_day, st_hour, end_month, end_day, end_hour)
    data  = data.filter_by_analysis_period(lb_ap)

    if conditional_statement:
        try:
            data = data.filter_by_conditional_statement(conditional_statement)
        except AssertionError:
            return "No values found for that conditional statement"
        except ValueError:
            return "Invalid conditional statement"

    if min:
        try: min = float(min)
        except ValueError: return "Invalid minimum value"
    if max:
        try: max = float(max)
        except ValueError: return "Invalid maximum value"

    lb_lp = LegendParameters(colors=colorsets[global_colorset])
    if min: lb_lp.min = min
    if max: lb_lp.max = max

    hp         = HourlyPlot(data, legend_parameters=lb_lp)
    var_name   = str(data.header.data_type)
    var_unit   = str(data.header.unit)
    return hp.plot(title=f"{var_name} ({var_unit})", show_title=True)


# ─────────────────────────────────────────────────────────────────────────────
# f313
# ─────────────────────────────────────────────────────────────────────────────
def f313__get_bar_chart_figure(
    fields: dict, epw: EPW, selection: List[str],
    data_type: str, switch: bool, stack: bool, global_colorset: str,
) -> Figure:
    """Monthly or daily bar chart for selected EPW fields."""
    colors = _get_colors(switch, global_colorset)
    data   = []
    for count, item in enumerate(selection):
        if item:
            var = epw._get_data_by_field(fields[list(fields.keys())[count]])
            if   data_type == "Monthly average": data.append(var.average_monthly())
            elif data_type == "Monthly total":   data.append(var.total_monthly())
            elif data_type == "Daily average":   data.append(var.average_daily())
            elif data_type == "Daily total":     data.append(var.total_daily())

    lb_lp = LegendParameters(colors=colors)
    return MonthlyChart(data, legend_parameters=lb_lp, stack=stack).plot(
        title=data_type, center_title=True)


# ─────────────────────────────────────────────────────────────────────────────
# f314
# ─────────────────────────────────────────────────────────────────────────────
def f314__get_hourly_line_chart_figure(
    data: HourlyContinuousCollection, switch: bool, global_colorset: str,
    font_family: str = "SansSerif", font_size: int = 12, margin: dict = None,
) -> Figure:
    """Customised hourly line chart from a HourlyContinuousCollection."""
    if margin is None:
        margin = dict(l=20, r=20, t=30, b=10)
    colors = _get_colors(switch, global_colorset)
    figure = data.line_chart(color=colors[-1], title=data.header.data_type.name, show_title=True)
    figure.update_layout(margin=margin, title=dict(x=0.5, xanchor="center"))
    if figure.layout.legend:
        figure.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center",
                                         y=-0.1, yanchor="top"))
    return figure


# ─────────────────────────────────────────────────────────────────────────────
# f315
# ─────────────────────────────────────────────────────────────────────────────
def f315__get_hourly_diurnal_average_chart_figure(
    data: HourlyContinuousCollection, switch: bool, global_colorset: str
) -> Figure:
    """Diurnal average chart from a HourlyContinuousCollection."""
    colors = _get_colors(switch, global_colorset)
    return data.diurnal_average_chart(title=data.header.data_type.name,
                                      show_title=True, color=colors[-1])


# ─────────────────────────────────────────────────────────────────────────────
# f316
# ─────────────────────────────────────────────────────────────────────────────
def f316__get_daily_chart_figure(
    data: HourlyContinuousCollection, switch: bool, global_colorset: str
) -> Figure:
    """Daily average bar chart from a HourlyContinuousCollection."""
    colors = _get_colors(switch, global_colorset)
    data   = data.average_daily()
    return data.bar_chart(color=colors[-1], title=data.header.data_type.name, show_title=True)


# ─────────────────────────────────────────────────────────────────────────────
# f317
# ─────────────────────────────────────────────────────────────────────────────
def f317__get_sunpath_figure(
    sunpath_type: str, global_colorset: str,
    epw: EPW = None, switch: bool = False,
    data: HourlyContinuousCollection = None,
) -> Figure:
    """Sunpath figure from EPW location."""
    lb_sunpath = Sunpath.from_location(epw.location)
    if sunpath_type == "from epw location":
        return lb_sunpath.plot(colorset=_get_colors(switch, global_colorset),
                               title=epw.location.city, show_title=True)
    return lb_sunpath.plot(colorset=colorsets[global_colorset], data=data,
                           title=data.header.data_type.name, show_title=True)


# ─────────────────────────────────────────────────────────────────────────────
# f318
# ─────────────────────────────────────────────────────────────────────────────
def f318__get_degree_days_figure(
    dbt: HourlyContinuousCollection, _heat_base_: int, _cool_base_: int,
    stack: bool, switch: bool, global_colorset: str,
) -> Tuple[Figure, HourlyContinuousCollection, HourlyContinuousCollection]:
    """Monthly HDD/CDD bar chart."""
    hourly_heat = HourlyContinuousCollection.compute_function_aligned(
        heating_degree_time, [dbt, _heat_base_], HeatingDegreeTime(), "degC-hours")
    hourly_heat.convert_to_unit("degC-days")

    hourly_cool = HourlyContinuousCollection.compute_function_aligned(
        cooling_degree_time, [dbt, _cool_base_], CoolingDegreeTime(), "degC-hours")
    hourly_cool.convert_to_unit("degC-days")

    lb_lp = LegendParameters(colors=_get_colors(switch, global_colorset))
    chart = MonthlyChart([hourly_cool.total_monthly(), hourly_heat.total_monthly()],
                         legend_parameters=lb_lp, stack=stack)
    return chart.plot(title="Degree Days", center_title=True), hourly_heat, hourly_cool


# ─────────────────────────────────────────────────────────────────────────────
# f319
# ─────────────────────────────────────────────────────────────────────────────
def f319__get_windrose_figure(
    st_month: int, st_day: int, st_hour: int,
    end_month: int, end_day: int, end_hour: int,
    epw: EPW, global_colorset: str,
) -> Figure:
    """Ladybug windrose figure filtered to the given analysis period."""
    lb_ap    = AnalysisPeriod(st_month, st_day, st_hour, end_month, end_day, end_hour)
    wind_dir = epw.wind_direction.filter_by_analysis_period(lb_ap)
    wind_spd = epw.wind_speed.filter_by_analysis_period(lb_ap)

    lb_lp        = LegendParameters(colors=colorsets[global_colorset])
    lb_wind_rose = WindRose(wind_dir, wind_spd)
    lb_wind_rose.legend_parameters = lb_lp
    return lb_wind_rose.plot(title="Wind-Rose", show_title=True)


# ─────────────────────────────────────────────────────────────────────────────
# f320
# ─────────────────────────────────────────────────────────────────────────────
def f320__get_psy_chart_figure(
    epw: EPW, global_colorset: str, selected_strategy: str,
    load_data: bool, draw_polygons: bool, data: HourlyContinuousCollection,
) -> Figure:
    """Psychrometric chart from EPW dry-bulb / RH data."""
    lb_lp  = LegendParameters(colors=colorsets[global_colorset])
    lb_psy = PsychrometricChart(epw.dry_bulb_temperature, epw.relative_humidity,
                                legend_parameters=lb_lp)

    strategy_map = {
        "All": [Strategy.comfort, Strategy.evaporative_cooling,
                Strategy.mas_night_ventilation, Strategy.occupant_use_of_fans,
                Strategy.capture_internal_heat, Strategy.passive_solar_heating],
        "Comfort":                [Strategy.comfort],
        "Evaporative Cooling":    [Strategy.evaporative_cooling],
        "Mass + Night Ventilation": [Strategy.mas_night_ventilation],
        "Occupant use of fans":   [Strategy.occupant_use_of_fans],
        "Capture internal heat":  [Strategy.capture_internal_heat],
        "Passive solar heating":  [Strategy.passive_solar_heating],
    }
    strategies = strategy_map.get(selected_strategy, [Strategy.comfort])
    pmv = PolygonPMV(lb_psy)

    if load_data:
        return (lb_psy.plot(data=data, polygon_pmv=pmv, strategies=strategies,
                            solar_data=epw.direct_normal_radiation)
                if draw_polygons else lb_psy.plot(data=data))
    return (lb_psy.plot(polygon_pmv=pmv, strategies=strategies,
                        solar_data=epw.direct_normal_radiation)
            if draw_polygons else lb_psy.plot())


# ─────────────────────────────────────────────────────────────────────────────
# f321
# ─────────────────────────────────────────────────────────────────────────────
def f321__line_chart_daily_range(
    hourly_data: pd.DataFrame, show_legend: bool, legend_position: str,
    margins: tuple, fixed_y_range, var_name: str, var_unit: str,
) -> Figure:
    """Daily min/max range with average overlay line chart."""
    daily = hourly_data.resample("D").agg(
        min_values=("values", "min"), max_values=("values", "max"),
        average_values=("values", "mean"),
    ).reset_index()

    trace1 = go.Scatter(x=daily["datetime"], y=daily["min_values"],
                        mode="lines", line=dict(width=0), showlegend=False)
    trace2 = go.Scatter(x=daily["datetime"], y=daily["max_values"],
                        fill="tonexty", mode="lines",
                        fillcolor="rgba(112,115,155,0.15)", line=dict(width=0),
                        name="Daily Range")
    trace3 = go.Scatter(x=daily["datetime"], y=daily["average_values"],
                        mode="lines", line=dict(color="#0a1585", width=2),
                        name="Daily Average")

    lp = {"top left": dict(x=0, y=1), "top right": dict(x=1, y=1),
          "bottom left": dict(x=0, y=0), "bottom right": dict(x=1, y=0)}

    layout = go.Layout(
        title=f"{var_name} - Min, Max, and Average",
        xaxis=dict(title="Date", tickformat="%b"),
        yaxis=dict(title=f"{var_name} [{var_unit}]",
                   range=fixed_y_range if fixed_y_range else None),
        showlegend=show_legend, legend=lp[legend_position],
        margin=dict(t=margins[0], r=margins[1], b=margins[2], l=margins[3]),
    )
    return go.Figure(data=[trace1, trace2, trace3], layout=layout)


# ─────────────────────────────────────────────────────────────────────────────
# f322
# ─────────────────────────────────────────────────────────────────────────────
def f322__plot_windrose(
    wind_direction, wind_speed, num_sectors: int = 16,
    speed_bins=None, unit: str = "m/s", color_map=None,
) -> Figure:
    """Custom Plotly polar windrose."""
    df = pd.DataFrame({"direction": wind_direction, "speed": wind_speed}).dropna()

    sector_size   = 360 / num_sectors
    sector_edges  = np.linspace(0, 360, num_sectors + 1)
    sector_labels = [(i + 0.5) * sector_size for i in range(num_sectors)]

    df["dir_bin"] = pd.cut(df["direction"] % 360, bins=sector_edges,
                           labels=sector_labels, include_lowest=True, right=False)

    if speed_bins is None:
        speed_bins = [0, 1, 3, 5, 7, 10, 15, 20, np.inf]
    speed_labels = [f"{speed_bins[i]}–{speed_bins[i+1]} {unit}"
                    for i in range(len(speed_bins) - 1)]
    df["speed_bin"] = pd.cut(df["speed"], bins=speed_bins,
                              labels=speed_labels, include_lowest=True)

    bin_mins = [float(label.split("–")[0]) for label in speed_labels]
    norm     = (np.array(bin_mins) - min(bin_mins)) / (max(bin_mins) - min(bin_mins))
    color_lookup = {label: _rgb_hex(color_map[int(x * (len(color_map) - 1))][:3])
                    for label, x in zip(speed_labels, norm)}

    grouped = (df.groupby(["dir_bin", "speed_bin"], observed=False).size()
               .reset_index(name="count"))
    grouped["frequency"] = 100 * grouped["count"] / grouped["count"].sum()

    wide_df = (grouped.pivot(index="dir_bin", columns="speed_bin", values="frequency")
               .fillna(0).reindex(sector_labels))

    fig = go.Figure()
    for speed_bin in wide_df.columns:
        fig.add_trace(go.Barpolar(
            r=wide_df[speed_bin], theta=wide_df.index.astype(float),
            name=speed_bin, marker_color=color_lookup.get(speed_bin, "#cccccc"),
        ))

    fig.update_layout(template="plotly_white",
                      polar=dict(angularaxis=dict(direction="clockwise", rotation=90),
                                 radialaxis=dict(ticksuffix="%", showticklabels=True)),
                      legend_title="Wind Speed", margin=dict(l=20, r=20, t=60, b=20))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# f323
# ─────────────────────────────────────────────────────────────────────────────
def f323__daytime_nighttime_scatter(
    df_daytime: pd.DataFrame, df_nighttime: pd.DataFrame
) -> Figure:
    """Scatter comparing daytime vs night-time temperature statistics by month."""
    def _prep(raw, cols):
        d = raw.T.reset_index()
        d.columns = ["Month"] + cols
        for c in cols:
            d[c] = pd.to_numeric(d[c])
        return d

    df_d = _prep(df_daytime,  ["DayTime Max",   "DayTime Min",   "DayTime Avg"])
    df_n = _prep(df_nighttime, ["NightTime Max", "NightTime Min", "NightTime Avg"])

    fig = go.Figure()
    for col, sym, sz, nm in [("DayTime Max", "square", 12, "Day Max"),
                               ("DayTime Min", "circle",  9, "Day Min"),
                               ("DayTime Avg", "diamond", 6, "Day Avg")]:
        fig.add_trace(go.Scatter(x=df_d["Month"], y=df_d[col], mode="markers",
                                 name=nm, marker=dict(color="gold", symbol=sym, size=sz)))
    for col, sym, sz, nm in [("NightTime Max", "square", 12, "Night Max"),
                               ("NightTime Min", "circle",  9, "Night Min"),
                               ("NightTime Avg", "diamond", 6, "Night Avg")]:
        fig.add_trace(go.Scatter(x=df_n["Month"], y=df_n[col], mode="markers",
                                 name=nm, marker=dict(color="darkblue", symbol=sym, size=sz)))

    fig.update_layout(xaxis_title="Month", yaxis_title="Temperature",
                      yaxis=dict(range=[-11, 41]),
                      margin=dict(l=10, r=10, t=60, b=20),
                      legend=dict(yanchor="top", y=1.25, xanchor="center",
                                  x=0.5, orientation="h"),
                      legend_title="Legend", template="plotly_white")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# f324
# ─────────────────────────────────────────────────────────────────────────────
def f324__plot_pv_monthly_comparison_subplots(
    df_plot: pd.DataFrame, tilt: float, azimuth: float,
    range_min_em: float, range_max_em: float,
    range_min_him: float, range_max_him: float,
    margin: dict,
) -> Optional[Figure]:
    """Side-by-side E_m (energy) and H(i)_m (irradiation) monthly bar charts."""
    try:
        if "E_m" not in df_plot.columns or "H(i)_m" not in df_plot.columns:
            raise ValueError("DataFrame must contain 'E_m' and 'H(i)_m' columns.")

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.15,
            subplot_titles=(
                f"Avg. sum of global irradiation [kWh/m²/mo]<br>Tilt:<b>{tilt}</b> - Azimuth:<b>{azimuth}</b>",
                f"Avg. energy production [kWh/mo]<br>Tilt:<b>{tilt}</b> - Azimuth:<b>{azimuth}</b>",
            ))

        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["E_m"],
                             name="E_m (kWh/mo)", marker_color="#3880C5"), row=1, col=2)
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["H(i)_m"],
                             name="H(i)_m (kWh/m²/mo)", marker_color="orange"), row=1, col=1)

        fig.update_layout(showlegend=False, height=450, width=1000,
                          margin=margin or dict(l=40, r=40, t=40, b=40))
        fig.update_annotations(font=dict(size=13, color="#333", family="Source Serif Pro, Georgia, serif"))
        fig.update_yaxes(title_text="kWh",    range=[range_min_em,  range_max_em],  row=1, col=2)
        fig.update_yaxes(title_text="kWh/m²", range=[range_min_him, range_max_him], row=1, col=1)
        fig.update_xaxes(title_text="Month", row=1, col=1)
        fig.update_xaxes(title_text="Month", row=1, col=2)
        return fig

    except ValueError as ve:
        st.error(f"ValueError: {ve}")
    except Exception as e:
        st.error(f"Error generating the chart: {e}")


# ── Backward-compatible aliases ───────────────────────────────────────────────
absrd_avg_daily_profile                  = f301__avg_daily_profile
plot_hourly_line_chart_with_slider       = f302__plot_hourly_line_chart_with_slider
convert_rgba_to_plotly_colorscale        = f303__convert_rgba_to_plotly_colorscale
plot_heatmap_from_datetime_index         = f304__plot_heatmap_from_datetime_index
slice_data_by_month                      = f305__slice_data_by_month
bin_timeseries_data                      = f306__bin_timeseries_data
normalize_data                           = f307__normalize_data
create_stacked_bar_chart                 = f308__create_stacked_bar_chart
get_figure_config                        = f309__get_figure_config
get_fields                               = f310__get_fields
get_diurnal_average_chart_figure         = f311__get_diurnal_average_chart_figure
get_hourly_data_figure                   = f312__get_hourly_data_figure
get_bar_chart_figure                     = f313__get_bar_chart_figure
get_hourly_line_chart_figure             = f314__get_hourly_line_chart_figure
get_hourly_diurnal_average_chart_figure  = f315__get_hourly_diurnal_average_chart_figure
get_daily_chart_figure                   = f316__get_daily_chart_figure
get_sunpath_figure                       = f317__get_sunpath_figure
get_degree_days_figure                   = f318__get_degree_days_figure
get_windrose_figure                      = f319__get_windrose_figure
get_psy_chart_figure                     = f320__get_psy_chart_figure
absrd__line_chart_daily__range           = f321__line_chart_daily_range
plot_windrose                            = f322__plot_windrose
absrd_daytime_nighttime_scatter          = f323__daytime_nighttime_scatter
plot_pv_monthly_comparison_subplots      = f324__plot_pv_monthly_comparison_subplots
