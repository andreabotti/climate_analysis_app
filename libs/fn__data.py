"""
libs/fn__data.py
================
Data layer: colour utilities, EPW helpers, PVGIS API, PVWatts V8 API,
and all data-processing / parsing functions.

Number range: 101–124  (data series)

Absorbs:
  - libs/fn__libraries.py  (data & EPW sections)
  - libs/fn_pvwatts.py

Function index
--------------
  f101__apply_analysis_period         – filter LB data by AnalysisPeriod
  f102__slice_df_analysis_period      – filter DataFrame by analysis period
  f103__get_colors                    – get (optionally reversed) colorset
  f104__rgb_to_hex                    – convert RGB tuple to hex string
  f105__epw_hash_func                 – Streamlit cache hash for EPW
  f106__hourly_data_hash_func         – Streamlit cache hash for HourlyContinuousCollection
  f107__color_hash_func               – Streamlit cache hash for Color
  f108__list_zip_files_in_s3          – list .zip files in an S3 bucket
  f109__download_file                 – download file from URL to disk
  f110__get_epw_from_url_or_path      – load EPW from URL or local path
  f111__add_to_epw_files_list         – append EPW entry to session-state list
  f112__filter_epw_object             – shallow-copy EPW filtered by period
  f113__process_stat_file             – parse .stat file → dry-bulb DataFrame
  f114__convert_outputs_to_dataframe  – PVGIS hourly outputs → DataFrame
  f115__fetch_pvgis_hourly_data       – hourly data from PVGIS API
  f116__fetch_pv_production_data      – monthly PV production from PVGIS API
  f117__parse_pv_production_data      – parse PVGIS monthly JSON → DataFrames
  f118__extract_meta_monthly_variable_info  – PVGIS meta variable bullets
  f119__extract_meta_variable_info_by_type  – filter PVGIS meta bullets by type
  f120__split_and_transpose_pv_monthly – split monthly DF into daily/monthly tables
  f121__iterate_pv_production         – sweep azimuth × tilt via PVGIS API
  f122__define_yaxis_range            – round min/max to nice axis bounds
  f123__get_api_key                   – retrieve NREL API key
  f124__pvwatts_query                 – cached PVWatts V8 API call
  f125__list_local_archive_epws       – browse local archive folder for all datasets for a station
  PVWattsResult                       – dataclass for PVWatts results
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.color import Color, Colorset
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW, EPWFields

# ── Colorset catalogue ────────────────────────────────────────────────────────
colorsets = {
    "original":               Colorset.original(),
    "nuanced":                Colorset.nuanced(),
    "annual_comfort":         Colorset.annual_comfort(),
    "benefit":                Colorset.benefit(),
    "benefit_harm":           Colorset.benefit_harm(),
    "black_to_white":         Colorset.black_to_white(),
    "blue_green_red":         Colorset.blue_green_red(),
    "cloud_cover":            Colorset.cloud_cover(),
    "cold_sensation":         Colorset.cold_sensation(),
    "ecotect":                Colorset.ecotect(),
    "energy_balance":         Colorset.energy_balance(),
    "energy_balance_storage": Colorset.energy_balance_storage(),
    "glare_study":            Colorset.glare_study(),
    "harm":                   Colorset.harm(),
    "heat_sensation":         Colorset.heat_sensation(),
    "multi_colored":          Colorset.multi_colored(),
    "multicolored_2":         Colorset.multicolored_2(),
    "multicolored_3":         Colorset.multicolored_3(),
    "openstudio_palette":     Colorset.openstudio_palette(),
    "peak_load_balance":      Colorset.peak_load_balance(),
    "shade_benefit":          Colorset.shade_benefit(),
    "shade_benefit_harm":     Colorset.shade_benefit_harm(),
    "shade_harm":             Colorset.shade_harm(),
    "shadow_study":           Colorset.shadow_study(),
    "therm":                  Colorset.therm(),
    "thermal_comfort":        Colorset.thermal_comfort(),
    "view_study":             Colorset.view_study(),
}

# ── PVWatts constants ─────────────────────────────────────────────────────────
PVWATTS_V8_URL  = "https://developer.nrel.gov/api/pvwatts/v8.json"
REQUEST_TIMEOUT = 30  # seconds

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ARRAY_TYPE_LABELS = {
    0: "Fixed – Open Rack",
    1: "Fixed – Roof Mount",
    2: "1-Axis Tracking",
    3: "1-Axis Backtracking",
    4: "2-Axis Tracking",
}

MODULE_TYPE_LABELS = {
    0: "Standard",
    1: "Premium",
    2: "Thin Film",
}


# ─────────────────────────────────────────────────────────────────────────────
# f101
# ─────────────────────────────────────────────────────────────────────────────
def f101__apply_analysis_period(data, plot_analysis_period):
    """
    Filter a Ladybug HourlyContinuousCollection by an analysis period list.

    Parameters
    ----------
    data : HourlyContinuousCollection
    plot_analysis_period : list[int]
        [start_month, start_day, start_hour, end_month, end_day, end_hour]
    """
    start_month, start_day, start_hour, end_month, end_day, end_hour = plot_analysis_period
    lb_ap = AnalysisPeriod(start_month, start_day, start_hour, end_month, end_day, end_hour)
    return data.filter_by_analysis_period(lb_ap)


# ─────────────────────────────────────────────────────────────────────────────
# f102
# ─────────────────────────────────────────────────────────────────────────────
def f102__slice_df_analysis_period(df: pd.DataFrame, plot_analysis_period) -> pd.DataFrame:
    """
    Slice a DataFrame with a DatetimeIndex according to an analysis period.
    Handles wrap-around periods (e.g. Oct–Feb).
    """
    start_month, start_day, start_hour, end_month, end_day, end_hour = plot_analysis_period

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df["datetime"])

    if (start_month > end_month) or (start_month == end_month and start_day > end_day):
        condition = (
            (df.index.month > start_month)
            | ((df.index.month == start_month) & (df.index.day >= start_day))
            | ((df.index.month == start_month) & (df.index.day == start_day) & (df.index.hour >= start_hour))
        ) | (
            (df.index.month < end_month)
            | ((df.index.month == end_month) & (df.index.day <= end_day))
            | ((df.index.month == end_month) & (df.index.day == end_day) & (df.index.hour <= end_hour))
        )
    else:
        condition = (
            (df.index.month > start_month)
            | ((df.index.month == start_month) & (df.index.day >= start_day))
            | ((df.index.month == start_month) & (df.index.day == start_day) & (df.index.hour >= start_hour))
        ) & (
            (df.index.month < end_month)
            | ((df.index.month == end_month) & (df.index.day <= end_day))
            | ((df.index.month == end_month) & (df.index.day == end_day) & (df.index.hour <= end_hour))
        )

    return df[condition]


# ─────────────────────────────────────────────────────────────────────────────
# f103
# ─────────────────────────────────────────────────────────────────────────────
def f103__get_colors(switch: bool, global_colorset: str) -> List[Color]:
    """
    Return a colorset, optionally reversed.

    Parameters
    ----------
    switch : bool
        Reverse the order when ``True``.
    global_colorset : str
        Key into the module-level ``colorsets`` dict.
    """
    if switch:
        colors = list(colorsets[global_colorset])
        colors.reverse()
        return colors
    return colorsets[global_colorset]


# ─────────────────────────────────────────────────────────────────────────────
# f104
# ─────────────────────────────────────────────────────────────────────────────
def f104__rgb_to_hex(rgb_tuple) -> str:
    """Convert an RGB tuple ``(r, g, b[, a])`` to a ``#rrggbb`` hex string."""
    return "#{:02x}{:02x}{:02x}".format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])


# ─────────────────────────────────────────────────────────────────────────────
# f105
# ─────────────────────────────────────────────────────────────────────────────
def f105__epw_hash_func(epw: EPW) -> str:
    """Streamlit cache hash helper for EPW objects."""
    return epw.location.city


# ─────────────────────────────────────────────────────────────────────────────
# f106
# ─────────────────────────────────────────────────────────────────────────────
def f106__hourly_data_hash_func(hourly_data: HourlyContinuousCollection) -> Tuple:
    """Streamlit cache hash helper for HourlyContinuousCollection objects."""
    return (hourly_data.header.data_type, hourly_data.average,
            hourly_data.min, hourly_data.max)


# ─────────────────────────────────────────────────────────────────────────────
# f107
# ─────────────────────────────────────────────────────────────────────────────
def f107__color_hash_func(color: Color) -> Tuple[float, float, float]:
    """Streamlit cache hash helper for Ladybug Color objects."""
    return color.r, color.g, color.b


# ─────────────────────────────────────────────────────────────────────────────
# f108
# ─────────────────────────────────────────────────────────────────────────────
def f108__list_zip_files_in_s3(bucket: str, prefix: str = "epw_files/ITA/") -> List[str]:
    """
    Return a paginated list of ``.zip`` file keys in an S3 bucket / prefix.
    Credentials read from ``st.secrets``; falls back to env / instance profile.
    """
    import boto3

    kwargs: dict = {}
    try:
        aws_key    = st.secrets.get("AWS_ACCESS_KEY_ID")
        aws_secret = st.secrets.get("AWS_SECRET_ACCESS_KEY")
        aws_region = (st.secrets.get("AWS_DEFAULT_REGION")
                      or st.secrets.get("AWS_REGION") or "eu-north-1")
        if aws_key and aws_secret:
            kwargs.update(aws_access_key_id=aws_key,
                          aws_secret_access_key=aws_secret,
                          region_name=aws_region)
        else:
            kwargs["region_name"] = aws_region
    except Exception:
        kwargs["region_name"] = "eu-north-1"

    s3 = boto3.client("s3", **kwargs)
    zip_files: List[str] = []

    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix or ""):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                if key.lower().endswith(".zip"):
                    zip_files.append(key)
        return zip_files
    except Exception as e:
        st.error(f"Failed to list S3 objects (bucket={bucket}, prefix={prefix}): {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# f109
# ─────────────────────────────────────────────────────────────────────────────
def f109__download_file(url: str, save_path) -> bool:
    """
    Download *url* and write to *save_path* (a ``pathlib.Path``).
    Returns ``True`` on success, ``False`` on HTTP error.
    """
    response = requests.get(url)
    if response.status_code == 200:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    st.error(f"Failed to download the EPW file from {url}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# f110
# ─────────────────────────────────────────────────────────────────────────────
def f110__get_epw_from_url_or_path(epw_file_path, url: str = None) -> Optional[EPW]:
    """
    Return an EPW object loaded from a local path or downloaded from *url*.
    If the file already exists on disk it is not re-downloaded.
    """
    if url:
        if not epw_file_path.exists():
            success = f109__download_file(url, epw_file_path)
            if not success:
                st.error(f"Unable to download EPW file from {url}")
                return None
    return EPW(epw_file_path)


# ─────────────────────────────────────────────────────────────────────────────
# f111
# ─────────────────────────────────────────────────────────────────────────────
def f111__add_to_epw_files_list(
    epw_file_name: str, epw_object: EPW, epw_file_path
) -> None:
    """
    Append an EPW entry to ``st.session_state['epw_files_list']``.
    Shows a warning if the file is already present.
    """
    existing = [item["epw_file_name"] for item in st.session_state["epw_files_list"]]
    if epw_file_name not in existing:
        st.session_state["epw_files_list"].append({
            "epw_file_name":  epw_file_name,
            "epw_object":     epw_object,
            "epw_file_path":  epw_file_path,
            "stat_file_name": epw_file_name.replace(".epw", ".stat"),
            "stat_file_path": str(epw_file_path).replace(".epw", ".stat"),
        })
        st.success(f"Added {epw_file_name} to Analysis List.")
    else:
        st.warning(f"{epw_file_name} is already in the analysis list.")


# ─────────────────────────────────────────────────────────────────────────────
# f112
# ─────────────────────────────────────────────────────────────────────────────
def f112__filter_epw_object(
    epw: EPW,
    start_month: int, start_day: int, start_hour: int,
    end_month: int,   end_day: int,   end_hour: int,
) -> EPW:
    """
    Return a shallow copy of *epw* with all hourly fields filtered to the
    given analysis period.
    """
    analysis_period = [start_month, start_day, start_hour, end_month, end_day, end_hour]
    filtered_epw = EPW.__new__(EPW)
    filtered_epw.location = epw.location
    filtered_epw.header   = epw.header

    for attr in [
        "dry_bulb_temperature", "dew_point_temperature", "relative_humidity",
        "atmospheric_station_pressure", "extraterrestrial_horizontal_radiation",
        "extraterrestrial_direct_normal_radiation", "horizontal_infrared_radiation_intensity",
        "global_horizontal_radiation", "direct_normal_radiation",
        "diffuse_horizontal_radiation", "global_horizontal_illuminance",
        "direct_normal_illuminance", "diffuse_horizontal_illuminance",
        "zenith_luminance", "wind_direction", "wind_speed",
        "total_sky_cover", "opaque_sky_cover", "visibility",
        "ceiling_height", "present_weather_observation", "present_weather_codes",
        "precipitable_water", "aerosol_optical_depth", "snow_depth",
        "days_since_last_snowfall", "albedo", "liquid_precipitation_depth",
        "liquid_precipitation_quantity",
    ]:
        setattr(filtered_epw, attr,
                f101__apply_analysis_period(getattr(epw, attr), analysis_period))

    return filtered_epw


# ─────────────────────────────────────────────────────────────────────────────
# f113
# ─────────────────────────────────────────────────────────────────────────────
def f113__process_stat_file(file_path: str):
    """
    Parse a ``.stat`` file and return a dry-bulb temperature DataFrame plus
    a summary dict.
    """
    with open(file_path, "r", encoding="ISO-8859-1") as fh:
        file_content = fh.readlines()

    section_marker  = "Monthly Statistics for Dry Bulb temperatures [C]"
    section_data    = []
    inside_section  = False
    refined_pattern = re.compile(r"\s*\t+\s*|\s{2,}|\s*\t\s*")

    for line in file_content:
        if section_marker in line:
            inside_section = True
            continue
        if inside_section:
            if "Extreme Dry Bulb temperatures" in line:
                inside_section = False
                continue
            row = [item.strip() for item in re.split(refined_pattern, line) if item.strip()]
            if row:
                section_data.append(row)

    columns = section_data[0] if section_data else []
    rows    = section_data[1:] if len(section_data) > 1 else []

    dry_bulb_df_final = pd.DataFrame()
    if columns and rows:
        dry_bulb_df_final = pd.DataFrame(rows, columns=["Metric"] + columns)
        dry_bulb_df_final.set_index("Metric", inplace=True)

    summary_dict: dict = {}
    for row in section_data[-2:]:
        if len(row) < 2:
            continue
        if "Maximum Dry Bulb temperature" in row[0]:
            summary_dict["Maximum Dry Bulb Temperature"] = row[1].strip()
        elif "Minimum Dry Bulb temperature" in row[0]:
            summary_dict["Minimum Dry Bulb Temperature"] = row[1].strip()

    if not dry_bulb_df_final.empty:
        dry_bulb_df_final = dry_bulb_df_final.iloc[:-2]

    return dry_bulb_df_final, summary_dict


# ─────────────────────────────────────────────────────────────────────────────
# f114
# ─────────────────────────────────────────────────────────────────────────────
def f114__convert_outputs_to_dataframe(outputs: dict) -> pd.DataFrame:
    """
    Convert the ``"hourly"`` field of a PVGIS ``outputs`` dict to a
    DataFrame indexed by datetime.  Returns an empty DataFrame on failure.
    """
    try:
        hourly_data = outputs.get("hourly", [])
        if not isinstance(hourly_data, list) or len(hourly_data) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(hourly_data)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M")
            df.set_index("time", inplace=True)

        return df.apply(pd.to_numeric, errors="ignore")
    except Exception as e:
        st.error(f"Error converting hourly data to DataFrame: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# f115
# ─────────────────────────────────────────────────────────────────────────────
def f115__fetch_pvgis_hourly_data(
    lat: float, lon: float, startyear: int, endyear: int,
    peakpower: float = 1, loss: float = 14,
    angle: float = 35, azimuth: float = 180,
) -> Optional[dict]:
    """Fetch hourly PV production data from PVGIS seriescalc."""
    api_url = "https://re.jrc.ec.europa.eu/api/seriescalc"
    params  = dict(lat=lat, lon=lon, outputformat="json",
                   startyear=startyear, endyear=endyear,
                   peakpower=peakpower, loss=loss, angle=angle, azimuth=azimuth)
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError:
        # seriescalc may return 400; monthly data from PVcalc still works
        pass
    except requests.exceptions.RequestException:
        pass
    except json.JSONDecodeError:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# f116
# ─────────────────────────────────────────────────────────────────────────────
def f116__fetch_pv_production_data(
    lat: float, lon: float, peakpower: float, loss: float,
    azimuth: float, tilt: float,
) -> Optional[dict]:
    """Fetch monthly PV production data from PVGIS PVcalc."""
    api_url = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"
    params  = dict(lat=lat, lon=lon, peakpower=peakpower, pvtechchoice="crystSi",
                   loss=loss, fixed=1, angle=tilt, aspect=azimuth, outputformat="json")
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# f117
# ─────────────────────────────────────────────────────────────────────────────
def f117__parse_pv_production_data(raw_json: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse PVGIS PVcalc JSON to monthly + totals DataFrames."""
    try:
        monthly_data = raw_json.get("outputs", {}).get("monthly", {}).get("fixed", [])
        df_monthly   = pd.DataFrame(monthly_data)
        totals_data  = raw_json.get("outputs", {}).get("totals", {}).get("fixed", {})
        df_totals    = pd.DataFrame([totals_data], index=["Yearly Totals"])
        return df_monthly, df_totals
    except Exception as e:
        st.error(f"Error parsing PV production data: {e}")
        return pd.DataFrame(), pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# f118
# ─────────────────────────────────────────────────────────────────────────────
def f118__extract_meta_monthly_variable_info(pv_production_data: dict) -> List[str]:
    """Extract Markdown bullets for PVGIS monthly variables."""
    try:
        monthly_variables = (pv_production_data
                             .get("meta", {}).get("outputs", {})
                             .get("monthly", {}).get("variables", {}))
        if monthly_variables:
            return [
                f"- **{name}**: {details.get('description', 'No description')} "
                f"(Units: {details.get('units', 'N/A')})"
                for name, details in monthly_variables.items()
            ]
        return ["- No monthly variables found in the 'meta' section."]
    except Exception as e:
        st.error(f"Error extracting meta monthly variable information: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# f119
# ─────────────────────────────────────────────────────────────────────────────
def f119__extract_meta_variable_info_by_type(
    pv_production_data: dict, daily_only: bool = False
) -> List[str]:
    """Filter PVGIS monthly meta bullets by daily vs monthly variables."""
    all_bullets = f118__extract_meta_monthly_variable_info(pv_production_data)
    if not all_bullets or all_bullets[0].startswith("- No monthly"):
        return all_bullets

    monthly_variables = (pv_production_data
                         .get("meta", {}).get("outputs", {})
                         .get("monthly", {}).get("variables", {}))
    filtered = []
    for var_name, var_details in monthly_variables.items():
        is_daily = var_name.endswith("_d")
        if (daily_only and is_daily) or (not daily_only and not is_daily):
            filtered.append(
                f"- **{var_name}**: {var_details.get('description', 'N/A')} "
                f"(Units: {var_details.get('units', 'N/A')})"
            )
    return filtered if filtered else ["- No variables of this type."]


# ─────────────────────────────────────────────────────────────────────────────
# f120
# ─────────────────────────────────────────────────────────────────────────────
def f120__split_and_transpose_pv_monthly(
    df_monthly: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split PV monthly DF into daily-var and monthly-var tables, transposed."""
    if df_monthly.empty or "month" not in df_monthly.columns:
        return pd.DataFrame(), pd.DataFrame()

    df           = df_monthly.set_index("month")
    daily_cols   = [c for c in df.columns if c.endswith("_d")]
    monthly_cols = [c for c in df.columns if c not in daily_cols]

    return (df[daily_cols].T  if daily_cols   else pd.DataFrame(),
            df[monthly_cols].T if monthly_cols else pd.DataFrame())


# ─────────────────────────────────────────────────────────────────────────────
# f121
# ─────────────────────────────────────────────────────────────────────────────
def f121__iterate_pv_production(
    lat: float, lon: float,
    azimuth_values: List[float], tilt_values: List[float],
    peakpower: float, loss: float, pause_duration: float,
) -> List[dict]:
    """Sweep azimuth × tilt combinations, calling PVGIS for each pair."""
    results = []
    for azimuth in azimuth_values:
        for tilt in tilt_values:
            result = f116__fetch_pv_production_data(lat, lon, peakpower, loss, azimuth, tilt)
            if result and "outputs" in result and "totals" in result["outputs"]:
                totals = result["outputs"]["totals"]["fixed"]
                results.append({
                    "azimuth": azimuth, "tilt": tilt,
                    "daily_energy":   totals.get("E_d"),
                    "monthly_energy": totals.get("E_m"),
                    "yearly_energy":  totals.get("E_y"),
                    "daily_irr":      totals.get("H(i)_d"),
                    "monthly_irr":    totals.get("H(i)_m"),
                    "yearly_irr":     totals.get("H(i)_y"),
                    "total_loss":     totals.get("l_total"),
                })
            time.sleep(pause_duration)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# f122
# ─────────────────────────────────────────────────────────────────────────────
def f122__define_yaxis_range(var_min: float, var_max: float) -> Tuple[float, float]:
    """Round min/max to 'nice' axis bounds based on data range."""
    range_diff = var_max - var_min
    if   range_diff <= 100:  round_base = 10
    elif range_diff <= 500:  round_base = 50
    elif range_diff <= 1000: round_base = 100
    else:                    round_base = 1000

    return (math.floor(var_min / round_base) * round_base,
            math.ceil(var_max  / round_base) * round_base)


# ─────────────────────────────────────────────────────────────────────────────
# f123
# ─────────────────────────────────────────────────────────────────────────────
def f123__get_api_key() -> Optional[str]:
    """Retrieve NREL API key from secrets/env."""
    key = None
    try:
        key = st.secrets.get("NREL_API_KEY") or st.secrets.get("nrel", {}).get("api_key")
    except Exception:
        pass
    if not key:
        key = os.environ.get("NREL_API_KEY")
    return key


# ── PVWatts result dataclass ──────────────────────────────────────────────────
@dataclass
class PVWattsResult:
    """Parsed, ready-to-use result from a PVWatts V8 API call."""

    lat: float; lon: float; system_capacity_kw: float
    tilt: float; azimuth: float; losses_pct: float
    array_type: int; module_type: int; dataset: str; timeframe: str

    ac_annual_kwh:  float = 0.0
    dc_annual_kwh:  float = 0.0
    solrad_annual:  float = 0.0

    ac_monthly_kwh: List[float] = field(default_factory=list)
    dc_monthly_kwh: List[float] = field(default_factory=list)
    poa_monthly:    List[float] = field(default_factory=list)
    solrad_monthly: List[float] = field(default_factory=list)

    ac_hourly_w:  List[float] = field(default_factory=list)
    poa_hourly:   List[float] = field(default_factory=list)
    tamb_hourly:  List[float] = field(default_factory=list)
    wspd_hourly:  List[float] = field(default_factory=list)

    station_name:   str   = ""
    station_elev_m: float = 0.0

    @property
    def specific_yield_kwh_kwp(self) -> float:
        """AC annual yield per kWp — a key LEED metric."""
        return self.ac_annual_kwh / self.system_capacity_kw if self.system_capacity_kw else 0.0

    @property
    def performance_ratio(self) -> float:
        """Simple PR: AC_annual / (POA_annual_kWh_m2 × capacity_kWp)."""
        poa_total = sum(self.poa_monthly)
        if poa_total and self.system_capacity_kw:
            return self.ac_annual_kwh / (poa_total * self.system_capacity_kw)
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# f124
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def f124__pvwatts_query(
    api_key: str, lat: float, lon: float,
    system_capacity_kw: float = 4.0, tilt: float = 25.0,
    azimuth: float = 180.0, losses_pct: float = 14.0,
    module_type: int = 0, array_type: int = 1,
    timeframe: str = "monthly", dataset: str = "intl",
) -> PVWattsResult:
    """Query NREL PVWatts V8 API and return a PVWattsResult."""
    params = dict(api_key=api_key, lat=lat, lon=lon,
                  system_capacity=system_capacity_kw, tilt=tilt, azimuth=azimuth,
                  losses=losses_pct, module_type=module_type, array_type=array_type,
                  timeframe=timeframe, dataset=dataset)

    resp = requests.get(PVWATTS_V8_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    body   = resp.json()
    errors = body.get("errors", [])
    if errors:
        raise ValueError(f"PVWatts API error: {'; '.join(errors)}")

    out = body.get("outputs", {})
    si  = body.get("station_info", {})

    return PVWattsResult(
        lat=lat, lon=lon, system_capacity_kw=system_capacity_kw,
        tilt=tilt, azimuth=azimuth, losses_pct=losses_pct,
        array_type=array_type, module_type=module_type,
        dataset=dataset, timeframe=timeframe,
        ac_annual_kwh  = float(out.get("ac_annual",  0)),
        dc_annual_kwh  = float(out.get("dc_annual",  0)),
        solrad_annual  = float(out.get("solrad_annual", 0)),
        ac_monthly_kwh = [float(v) for v in out.get("ac_monthly",     [])],
        dc_monthly_kwh = [float(v) for v in out.get("dc_monthly",     [])],
        poa_monthly    = [float(v) for v in out.get("poa_monthly",    [])],
        solrad_monthly = [float(v) for v in out.get("solrad_monthly", [])],
        ac_hourly_w  = [float(v) for v in out.get("ac",   [])],
        poa_hourly   = [float(v) for v in out.get("poa",  [])],
        tamb_hourly  = [float(v) for v in out.get("tamb", [])],
        wspd_hourly  = [float(v) for v in out.get("wspd", [])],
        station_name   = si.get("city", si.get("location", "")),
        station_elev_m = float(si.get("elev", 0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# f125  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

_TMYX_PERIOD_RE = re.compile(r"TMYx\.(\d{4})-(\d{4})", re.IGNORECASE)

def _extract_tmyx_period_label(filename: str) -> str:
    m = _TMYX_PERIOD_RE.search(filename)
    if m:
        return f"TMYx {m.group(1)}–{m.group(2)}"
    if "tmyx" in filename.lower():
        return "TMYx (unspecified)"
    return "EPW"

@dataclass
class ArchiveEpwEntry:
    """
    Represents one dataset in the local archive folder.
    """
    epw_path: "os.PathLike | str"
    stat_path: Optional["os.PathLike | str"]
    file_name: str
    period_label: str
    wmo: Optional[str]
    city: Optional[str]

def f125__list_local_archive_epws(
    local_root: str,
    country: str,
    state: str,
    city: str,
    wmo: Optional[str] = None,
) -> List[ArchiveEpwEntry]:
    """
    Scan ./data/<country>/<state>/ and return *all* EPW datasets matching the station.

    Matching strategy:
      - require city token in filename (case-insensitive), tolerant of '_' vs '-' vs spaces
      - if WMO is provided, additionally require '.<WMO>_' or '.<WMO>.' in filename

    This solves the issue where multiple TMYx variants exist for the same location
    (e.g. TMYx.2004-2018, TMYx.2007-2021, TMYx.2009-2023) but only one is shown.
    """
    base_dir = os.path.join(local_root, country, state)
    if not os.path.isdir(base_dir):
        return []

    # normalize city token to compare against normalized filename
    city_token = re.sub(r"[\s_]+", "-", city.strip()).lower()
    wmo_token = str(wmo).strip() if wmo else None

    results: List[ArchiveEpwEntry] = []
    for name in sorted(os.listdir(base_dir)):
        if not name.lower().endswith(".epw"):
            continue

        name_low = name.lower()
        norm_name = re.sub(r"[_\s]+", "-", name_low)  # filename normalization

        if city_token not in norm_name:
            continue

        if wmo_token:
            if (f".{wmo_token}_" not in name_low) and (f".{wmo_token}." not in name_low):
                continue

        epw_path = os.path.join(base_dir, name)
        stat_path = os.path.splitext(epw_path)[0] + ".stat"
        stat_path = stat_path if os.path.isfile(stat_path) else None

        results.append(
            ArchiveEpwEntry(
                epw_path=epw_path,
                stat_path=stat_path,
                file_name=name,
                period_label=_extract_tmyx_period_label(name),
                wmo=wmo_token,
                city=city,
            )
        )

    return results


# ── Backward-compatible aliases ───────────────────────────────────────────────
absrd_apply_analysis_period         = f101__apply_analysis_period
absrd_slice_df_analysis_period      = f102__slice_df_analysis_period
get_colors                          = f103__get_colors
rgb_to_hex                          = f104__rgb_to_hex
epw_hash_func                       = f105__epw_hash_func
hourly_data_hash_func               = f106__hourly_data_hash_func
color_hash_func                     = f107__color_hash_func
list_zip_files_in_s3                = f108__list_zip_files_in_s3
download_file                       = f109__download_file
get_epw_from_url_or_path            = f110__get_epw_from_url_or_path
add_to_epw_files_list               = f111__add_to_epw_files_list
filter_epw_object                   = f112__filter_epw_object
absrd_process_stat_file             = f113__process_stat_file
convert_outputs_to_dataframe        = f114__convert_outputs_to_dataframe
fetch_pvgis_hourly_data             = f115__fetch_pvgis_hourly_data
fetch_pv_production_data            = f116__fetch_pv_production_data
parse_pv_production_data            = f117__parse_pv_production_data
extract_meta_monthly_variable_info  = f118__extract_meta_monthly_variable_info
extract_meta_variable_info_by_type  = f119__extract_meta_variable_info_by_type
split_and_transpose_pv_monthly      = f120__split_and_transpose_pv_monthly
iterate_pv_production               = f121__iterate_pv_production
define_yaxis_range                  = f122__define_yaxis_range
get_api_key                         = f123__get_api_key
pvwatts_query                       = f124__pvwatts_query

# new helpers
list_local_archive_epws             = f125__list_local_archive_epws