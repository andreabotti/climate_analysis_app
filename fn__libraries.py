import os, tempfile
from io import StringIO
from typing import List, Tuple

import pandas as pd, numpy as np
from math import ceil, floor
from datetime import datetime, timedelta


import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

import streamlit as st
from streamlit_super_slider import st_slider

from ladybug_charts.to_figure import heat_map


from ladybug.epw import EPW
from ladybug.datatype.temperaturetime import HeatingDegreeTime, CoolingDegreeTime
from ladybug_comfort.degreetime import heating_degree_time, cooling_degree_time
from ladybug.datacollection import HourlyContinuousCollection
from ladybug_comfort.chart.polygonpmv import PolygonPMV
from ladybug.epw import EPW, EPWFields
from ladybug.color import Colorset, Color
from ladybug.legend import LegendParameters
from ladybug.hourlyplot import HourlyPlot
from ladybug.monthlychart import MonthlyChart
from ladybug.sunpath import Sunpath
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.windrose import WindRose
from ladybug.psychchart import PsychrometricChart
from ladybug_charts.utils import Strategy




colorsets = {
    'original': Colorset.original(),
    'nuanced': Colorset.nuanced(),
    'annual_comfort': Colorset.annual_comfort(),
    'benefit': Colorset.benefit(),
    'benefit_harm': Colorset.benefit_harm(),
    'black_to_white': Colorset.black_to_white(),
    'blue_green_red': Colorset.blue_green_red(),
    'cloud_cover': Colorset.cloud_cover(),
    'cold_sensation': Colorset.cold_sensation(),
    'ecotect': Colorset.ecotect(),
    'energy_balance': Colorset.energy_balance(),
    'energy_balance_storage': Colorset.energy_balance_storage(),
    'glare_study': Colorset.glare_study(),
    'harm': Colorset.harm(),
    'heat_sensation': Colorset.heat_sensation(),
    'multi_colored': Colorset.multi_colored(),
    'multicolored_2': Colorset.multicolored_2(),
    'multicolored_3': Colorset.multicolored_3(),
    'openstudio_palette': Colorset.openstudio_palette(),
    'peak_load_balance': Colorset.peak_load_balance(),
    'shade_benefit': Colorset.shade_benefit(),
    'shade_benefit_harm': Colorset.shade_benefit_harm(),
    'shade_harm': Colorset.shade_harm(),
    'shadow_study': Colorset.shadow_study(),
    'therm': Colorset.therm(),
    'thermal_comfort': Colorset.thermal_comfort(),
    'view_study': Colorset.view_study()
}



def get_colors(switch: bool, global_colorset: str) -> List[Color]:
    """Get switched colorset if requested.

    Args:
        switch: Boolean to switch colorset.
        global_colorset: Global colorset to use.

    Returns:
        List of colors.
    """

    if switch:
        colors = list(colorsets[global_colorset])
        colors.reverse()
    else:
        colors = colorsets[global_colorset]
    return colors


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def rgb_to_hex(rgb_tuple):
    return '#{:02x}{:02x}{:02x}'.format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])





def custom_hr():
    # Inject custom CSS to reduce vertical spacing before and after the horizontal line
    st.markdown("""
        <style>
            .hr-line {
                margin-top: -5px;
                margin-bottom: 0px;
            }
        </style>
    """, unsafe_allow_html=True)
    # Adding the horizontal line with reduced vertical spacing
    st.markdown('<hr class="hr-line">', unsafe_allow_html=True)





def epw_hash_func(epw: EPW) -> str:
    """Function to help streamlit hash an EPW object."""
    return epw.location.city

def hourly_data_hash_func(hourly_data: HourlyContinuousCollection) -> str:
    """Function to help streamlit hash an HourlyContinuousCollection object."""
    return hourly_data.header.data_type, hourly_data.average, hourly_data.min, hourly_data.max


def color_hash_func(color: Color) -> Tuple[float, float, float]:
    """Function to help streamlit hash a Color object."""
    return color.r, color.g, color.b
