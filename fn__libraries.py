import os, re, tempfile, requests
from io import StringIO
from typing import List, Tuple

import pandas as pd, numpy as np
from math import ceil, floor
from datetime import datetime, timedelta


import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
import pydeck as pdk

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




def absrd__epw_location_map(data, col_lat, col_lon, zoom_level, chart_height, chart_width):


    # Define the PyDeck Layer
    layer = pdk.Layer(
        'ScatterplotLayer',
        data,
        # get_position='[lon, lat]',
        get_position=f'[{col_lon}, {col_lat}]',
        get_radius=100,
        get_color='[255, 0, 0, 160]',
        pickable=True,
    )

    # Set the PyDeck View
    view_state = pdk.ViewState(
        latitude=data[f'{col_lat}'].mean(),
        longitude=data[f'{col_lon}'].mean(),
        zoom=zoom_level,
        pitch=0,
    )

    # Configure the map using st.pydeck_chart
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/light-v9',
            height=chart_height,  # Set the height of the map
            width=chart_width,   # Set the width of the map
        ),
    )











# Function to convert the hourly outputs to a DataFrame
def convert_outputs_to_dataframe(outputs):
    """
    Convert the 'hourly' field of the 'outputs' section of the PVGIS response to a Pandas DataFrame.
    
    Parameters:
    - outputs (dict): The 'outputs' section from PVGIS API response.
    
    Returns:
    - df (pd.DataFrame): DataFrame containing the processed data with 'time' as index.
    """
    try:
        # Extract 'hourly' data from outputs
        hourly_data = outputs.get('hourly', [])
        
        # If hourly data is not a list, return an empty DataFrame
        if not isinstance(hourly_data, list) or len(hourly_data) == 0:
            st.error("No hourly data found in the 'outputs' section.")
            return pd.DataFrame()

        # Convert hourly data to a DataFrame
        df = pd.DataFrame(hourly_data)

        # Convert 'time' column to datetime format and set it as index
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
            df.set_index('time', inplace=True)

        # Ensure other columns are numeric if possible
        df = df.apply(pd.to_numeric, errors='ignore')

        return df
    except Exception as e:
        st.error(f"Error converting hourly data to DataFrame: {e}")
        return pd.DataFrame()









# Function to fetch hourly PVGIS data
def fetch_pvgis_hourly_data(lat, lon, startyear, endyear, peakpower=1, loss=14, angle=35, azimuth=180):
    """
    Fetch hourly PVGIS data for the given latitude and longitude with specified parameters.
    """
    api_url = "https://re.jrc.ec.europa.eu/api/seriescalc"
    params = {
        'lat': lat,
        'lon': lon,
        'outputformat': 'json',
        'startyear': startyear,
        'endyear': endyear,
        'peakpower': peakpower,  # kWp
        'loss': loss,            # System losses in %
        'angle': angle,          # Tilt angle
        'azimuth': azimuth,      # Orientation
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Request error occurred: {req_err}")
    except json.JSONDecodeError as json_err:
        st.error(f"Error decoding JSON: {json_err}")
    
    return None










# Function to fetch PV production data
def fetch_pv_production_data(lat, lon, peakpower=1, loss=14):
    api_url = "https://re.jrc.ec.europa.eu/api/PVcalc"
    params = {
        'lat': lat,
        'lon': lon,
        'peakpower': peakpower,  # kWp
        'loss': loss,            # System losses in %
        'outputformat': 'json',
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        return response.json()  # Return JSON data directly since it is structured
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Request error occurred: {req_err}")
    
    return None







def parse_pv_production_data(raw_json):
    """
    Parse the raw JSON response from PV production API into structured DataFrames for monthly and total outputs.
    
    Parameters:
    - raw_json (dict): The JSON response from the API.
    
    Returns:
    - df_monthly (pd.DataFrame): DataFrame containing monthly PV production data.
    - df_totals (pd.DataFrame): DataFrame containing yearly totals data.
    """
    try:
        # Extract monthly data
        monthly_data = raw_json.get('outputs', {}).get('monthly', {}).get('fixed', [])
        df_monthly = pd.DataFrame(monthly_data)
        
        # Extract totals data
        totals_data = raw_json.get('outputs', {}).get('totals', {}).get('fixed', {})
        df_totals = pd.DataFrame([totals_data], index=["Yearly Totals"])

        return df_monthly, df_totals
    except Exception as e:
        st.error(f"Error parsing PV production data: {e}")
        return pd.DataFrame(), pd.DataFrame()












def extract_meta_monthly_variable_info(pv_production_data):
    """
    Extracts a list of monthly variable descriptions and units from the 'meta' section of PV production data.

    Parameters:
    - pv_production_data (dict): The raw JSON response containing PV production data.

    Returns:
    - bullet_points (list): List of bullet points summarizing the variable descriptions and units for the monthly data.
    """
    try:
        # Extract the 'meta' section and variables details for monthly data
        monthly_variables = pv_production_data.get('meta', {}).get('outputs', {}).get('monthly', {}).get('variables', {})

        # Create bullet-point list for monthly variables
        bullet_points = []
        if monthly_variables:
            for var_name, var_details in monthly_variables.items():
                description = var_details.get('description', 'No description available')
                units = var_details.get('units', 'No units available')
                bullet_points.append(f"- **{var_name}**: {description} (Units: {units})")
        else:
            bullet_points.append("- No monthly variables found in the 'meta' section.")

        return bullet_points

    except Exception as e:
        st.error(f"Error extracting meta monthly variable information: {e}")
        return []




def plot_pv_monthly_comparison_subplots(df_plot, range_min_em, range_max_em, range_min_him, range_max_him, margin):
    """
    Plots two subplots side-by-side using Plotly for E_m and H(i)_m from the given monthly PV DataFrame.
    Allows the user to specify y-axis ranges for both metrics separately and set custom plot margins.

    Parameters:
    - df_plot (pd.DataFrame): The DataFrame containing monthly PV production data.
    - range_min_em (float): Minimum value for the y-axis of E_m subplot.
    - range_max_em (float): Maximum value for the y-axis of E_m subplot.
    - range_min_him (float): Minimum value for the y-axis of H(i)_m subplot.
    - range_max_him (float): Maximum value for the y-axis of H(i)_m subplot.
    - margin (dict): Dictionary to set plot margins with keys 'l', 'r', 't', 'b' for left, right, top, and bottom.

    Returns:
    - None
    """
    try:
        # Check if the necessary columns are present in the DataFrame
        if 'E_m' not in df_plot.columns or 'H(i)_m' not in df_plot.columns:
            raise ValueError("The DataFrame must contain 'E_m' and 'H(i)_m' columns.")

        title_Em = 'Avg. energy production<br>[kWh/mo]'
        title_Him = 'Avg. sum of global irradiation<br>[kWh/m²/mo]'

        # Create a subplot with 1 row and 2 columns
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(title_Him, title_Em),
            horizontal_spacing=0.15,
        )

        # Add bar chart for E_m on the left subplot
        fig.add_trace(
            go.Bar(
                x=df_plot.index,
                y=df_plot['E_m'],
                name='E_m (kWh/mo)',
                marker_color='skyblue'
            ),
            row=1, col=2
        )

        # Add bar chart for H(i)_m on the right subplot
        fig.add_trace(
            go.Bar(
                x=df_plot.index,
                y=df_plot['H(i)_m'],
                name='H(i)_m (kWh/m²/mo)',
                marker_color='orange'
            ),
            row=1, col=1
        )

        # Update layout for both subplots
        fig.update_layout(
            # title='Monthly PV Production Comparison: E_m and H(i)_m',
            showlegend=False,  # Hide legend as each subplot has its own axis title
            height=500,        # Adjust the height of the chart
            width=1000,         # Adjust the width of the chart
            margin=margin if margin else dict(l=40, r=40, t=40, b=40),  # Apply user-defined margins or default
        )

        # Set custom y-axis ranges for each subplot
        fig.update_yaxes(
            title_text='kWh',
            range=[range_min_em, range_max_em], row=1, col=2)  # y-axis range for E_m
        fig.update_yaxes(
            title_text='(kWh/m²',
            range=[range_min_him, range_max_him], row=1, col=1)  # y-axis range for H(i)_m

        # Set x-axis title
        fig.update_xaxes(title_text='Month', row=1, col=1)
        fig.update_xaxes(title_text='Month', row=1, col=2)

        return fig

    except ValueError as ve:
        st.error(f"ValueError: {ve}")
    except Exception as e:
        st.error(f"Error generating the chart: {e}")








def absrd_style_df_streamlit(df, range_min_max, colormap):

    # Convert non-numeric values to NaN
    numeric_df = df.apply(pd.to_numeric, errors='coerce')

    if not range_min_max:
        range_min, range_max = numeric_df.min().min(), numeric_df.max().max()

    else:
        range_min = range_min_max[0]
        range_max = range_min_max[1]


    # Apply the gradient coloring using pandas Styler, only for numeric values
    styled_df = numeric_df.style.background_gradient(
        cmap=colormap,    # You can choose different colormaps like 'viridis', 'plasma', etc.
        axis=None,          # Apply to all cells
        vmin=range_min,     # Minimum value for the gradient
        vmax=range_max,      # Maximum value for the gradient
    ).format(precision=1)  # Set the decimal precision for display

    return styled_df



# Define a custom styling function to style odd or even cells
def highlight_odd_even_cells(val):
    """Apply bold and different colors to odd or even cells."""
    if isinstance(val, (int, float)):  # Check for numerical values
        # Make odd-indexed cells bold and change their color
        return 'font-weight: bold; color: blue;' if val % 2 != 0 else 'color: green;'
    else:
        return ''  # No styling for non-numeric values



# Define a function to style rows based on their position
def row_style(df, row_index):
    """Apply bold to rows 1 and 3, and light grey to other rows."""
    if row_index in [0, 2]:
        return ['font-weight: bold;'] * len(df.columns)
    else:
        return ['color: #65b3aa;'] * len(df.columns)





def absrd_daytime_nighttime_scatter(df_daytime, df_nighttime):

    # Transpose data for easier plotting and set the column names
    df_daytime = df_daytime.T.reset_index()
    df_daytime.columns = ['Month', 'DayTime Max', 'DayTime Min', 'DayTime Avg']

    df_nighttime = df_nighttime.T.reset_index()
    df_nighttime.columns = ['Month', 'NightTime Max', 'NightTime Min', 'NightTime Avg']

    # Convert y-axis columns to numeric data types
    df_daytime[['DayTime Max', 'DayTime Min', 'DayTime Avg']] = df_daytime[['DayTime Max', 'DayTime Min', 'DayTime Avg']].apply(pd.to_numeric)
    df_nighttime[['NightTime Max', 'NightTime Min', 'NightTime Avg']] = df_nighttime[['NightTime Max', 'NightTime Min', 'NightTime Avg']].apply(pd.to_numeric)


    # Create a scatter plot for each metric (Max, Min, Avg) for both DayTime and NightTime
    fig = go.Figure()

    # Add DayTime scatter points
    fig.add_trace(go.Scatter(
        x=df_daytime['Month'], y=df_daytime['DayTime Max'],
        mode='markers',
        name='Day Max',
        marker=dict(color='gold', symbol='square', size=12)
    ))

    fig.add_trace(go.Scatter(
        x=df_daytime['Month'], y=df_daytime['DayTime Min'],
        mode='markers',
        name='Day Min',
        marker=dict(color='gold', symbol='circle', size=9)
    ))

    fig.add_trace(go.Scatter(
        x=df_daytime['Month'], y=df_daytime['DayTime Avg'],
        mode='markers',
        name='Day Avg',
        marker=dict(color='gold', symbol='diamond', size=6)
    ))

    # Add NightTime scatter points
    fig.add_trace(go.Scatter(
        x=df_nighttime['Month'], y=df_nighttime['NightTime Max'],
        mode='markers',
        name='Night Max',
        marker=dict(color='darkblue', symbol='square', size=12)
    ))

    fig.add_trace(go.Scatter(
        x=df_nighttime['Month'], y=df_nighttime['NightTime Min'],
        mode='markers',
        name='Night Min',
        marker=dict(color='darkblue', symbol='circle', size=9)
    ))

    fig.add_trace(go.Scatter(
        x=df_nighttime['Month'], y=df_nighttime['NightTime Avg'],
        mode='markers',
        name='Night Avg',
        marker=dict(color='darkblue', symbol='diamond', size=6)
    ))


    # Update layout with y-axis range and margins
    fig.update_layout(
        # title='DayTime and NightTime Values by Month',
        xaxis_title='Month',
        yaxis_title='Temperature',
        yaxis=dict(range=[-11, 41]),  # Set the y-axis range from -10 to 40 (adjust as needed)
        margin=dict(l=10, r=10, t=60, b=20),  # Set margins (left, right, top, bottom)
 
        legend=dict(
            yanchor="top",
            y=1.25,  # Position the legend higher
            xanchor="center",
            x=0.5,  # Position the legend in the center horizontally
            orientation='h'  # Make the legend horizontal
        ),   
        legend_title='Legend',
        template='plotly_white'
    )


    return fig





def absrd_process_stat_file(file_path):

    # Read the contents of the uploaded file with ISO-8859-1 encoding to handle non-UTF-8 characters
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        file_content = file.readlines()


    # Initialize variables to capture the section lines
    section_marker = "Monthly Statistics for Dry Bulb temperatures [C]"
    section_data = []
    inside_dry_bulb_section = False

    # Use a regular expression to handle varied separators
    refined_pattern = re.compile(r'\s*\t+\s*|\s{2,}|\s*\t\s*')  # Pattern for tabs and multiple spaces

    # Start capturing lines after the section marker and continue until a completely new section starts
    for line in file_content:
        if section_marker in line:
            inside_dry_bulb_section = True
            continue

        if inside_dry_bulb_section:
            # Stop capturing only if a completely new section starts, not just for empty lines
            if "Extreme Dry Bulb temperatures" in line:
                inside_dry_bulb_section = False
                continue

            # Capture and process lines using the refined pattern, ignoring empty lines if necessary
            row = [item.strip() for item in re.split(refined_pattern, line) if item.strip()]
            if row:
                section_data.append(row)


    # Create a DataFrame from the captured data
    columns = section_data[0] if section_data else []  # Use the first row as column headers (months)
    rows = section_data[1:] if len(section_data) > 1 else []  # The remaining rows are metrics

    # Create the DataFrame if there is enough data
    if columns and rows:
        dry_bulb_df_final = pd.DataFrame(rows, columns=['Metric'] + columns)
        dry_bulb_df_final.set_index('Metric', inplace=True)

    # Extract the last two rows (summary rows) into a dictionary
    summary_rows = section_data[-2:]  # The last two rows are the summary rows
    summary_dict = {}

    # Extract the values from the summary rows
    for row in summary_rows:
        if "Maximum Dry Bulb temperature" in row[0]:
            summary_dict["Maximum Dry Bulb Temperature"] = row[1].strip()
        elif "Minimum Dry Bulb temperature" in row[0]:
            summary_dict["Minimum Dry Bulb Temperature"] = row[1].strip()

    # Remove the last two rows from the DataFrame
    dry_bulb_df_final = dry_bulb_df_final.iloc[:-2]


    return dry_bulb_df_final, summary_dict
