# IMPORT LIBRARIES
import os
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
from ladybug.epw import EPW

from fn__libraries import *
from fn__chart_libraries import *

mapbox_access_token = 'pk.eyJ1IjoiYW5kcmVhYm90dGkiLCJhIjoiY2xuNDdybms2MHBvMjJqbm95aDdlZ2owcyJ9.-fs8J1enU5kC3L4mAJ5ToQ'

####################################################################################
# PAGE HEADER
from fn__page_header import create_page_header
create_page_header()

####################################################################################
# LOAD DATA INTO SESSION STATE
epw_files_list = st.session_state['epw_files_list']
epw_files = [item['epw_object'] for item in epw_files_list]
epw_names = [item['epw_file_name'] for item in epw_files_list]
st.session_state['epw_names'] = epw_names
st.session_state['epw_files'] = epw_files

# Function to check if required keys exist in session state
if len(epw_names) < 1:
    st.warning("EPW files are not uploaded. Please go to the 'Data Upload' page to upload your EPW files.")
    st.stop()

####################################################################################
with st.sidebar:

    # Introduced checkboxes to plot charts
    fields = get_fields()
    hourly_selected = st.selectbox('Select an environmental variable', options=fields.keys(), key='data')
    global_colorset = st.selectbox('Select a color set', list(colorsets.keys()), index=1)

    custom_hr()

    ##########
    st.markdown("#### Filter data by date/time and values")
    with st.expander(label='Filter data - Date / Time'):
        sub_col_1, sub_col_2 = st.columns([1, 1])
        data_start_month = sub_col_1.number_input('Start month', min_value=1, max_value=12, value=1, key='data_start_month')
        data_end_month = sub_col_2.number_input('End month', min_value=1, max_value=12, value=12, key='data_end_month')
        data_start_day = sub_col_1.number_input('Start day', min_value=1, max_value=31, value=1, key='data_start_day')
        data_end_day = sub_col_2.number_input('End day', min_value=1, max_value=31, value=31, key='data_end_day')
        data_start_hour = sub_col_1.number_input('Start hour', min_value=0, max_value=23, value=0, key='data_start_hour')
        data_end_hour = sub_col_2.number_input('End hour', min_value=0, max_value=23, value=23, key='data_end_hour')

    with st.expander(label='Filter data - Values'):
        sub_col_1, sub_col_2 = st.columns([1, 1])
        data_min = sub_col_1.text_input('Min Value')
        data_max = sub_col_2.text_input('Max Value')
        data_conditional_statement = st.text_input('Apply conditional statement', help=help_text_01)

    custom_hr()

####################################################################################
# Replace Ladybug Analysis Period with Pandas Filtering
all_values = []
chart_cols = st.columns(len(epw_names))

for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
    with col:
        # Extract the data and timestamps
        data_col = epw_col._get_data_by_field(fields[hourly_selected])
        data_values = data_col.values
        data_dates = data_col.datetimes

        # Get year from the data for dynamic filtering
        data_year = data_dates[0].year if data_dates else 2023
        start_datetime = pd.Timestamp(year=data_year, month=data_start_month, day=data_start_day, hour=data_start_hour)
        end_datetime = pd.Timestamp(year=data_year, month=data_end_month, day=data_end_day, hour=data_end_hour)

        # Create a Pandas DataFrame
        df = pd.DataFrame({"datetime": data_dates, "values": data_values})
        df.set_index("datetime", inplace=True)  # Set datetime as the index for filtering

        # Apply filtering based on the analysis period
        filtered_df = df.loc[start_datetime:end_datetime]

        # Further filtering by min/max values if specified
        if data_min:
            filtered_df = filtered_df[filtered_df["values"] >= float(data_min)]
        if data_max:
            filtered_df = filtered_df[filtered_df["values"] <= float(data_max)]

        # Extract filtered values
        if not filtered_df.empty:
            all_values.append(filtered_df["values"].max())
            all_values.append(filtered_df["values"].min())
        else:
            all_values.append(0)  # Default value
            all_values.append(0)

# Validate all_values
if all_values:
    min_value = min(all_values)
    max_value = max(all_values)
else:
    min_value = 0
    max_value = 1  # Adjust default range as needed

# Define axis range
axis_min, axis_max = define_yaxis_range(var_min=min_value, var_max=max_value)
axis_min = axis_min * 1.05
axis_max = axis_max * 1.05

####################################################################################
# Tabs for Hourly and Daily Charts
tabs = st.tabs([
    "Hourly - **Line Chart with Plotly**",
    'VOID',
    "Hourly - Pandas - **Heatmap**",
    "Hourly - **Avg. Day**",
])

########################################
# Hourly Chart - LINE
with tabs[0]:
    st.markdown(f'##### Line chart with hourly data')
    st.write('')

    chart_cols = st.columns(len(epw_names))
    for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
        with col:
            st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_name}</h6>", unsafe_allow_html=True)

            # Extract the filtered DataFrame
            data_col = epw_col._get_data_by_field(fields[hourly_selected])
            data_values = data_col.values
            data_dates = data_col.datetimes

            # Get year from the data for dynamic filtering
            data_year = data_dates[0].year if data_dates else 2023
            start_datetime = pd.Timestamp(year=data_year, month=data_start_month, day=data_start_day, hour=data_start_hour)
            end_datetime = pd.Timestamp(year=data_year, month=data_end_month, day=data_end_day, hour=data_end_hour)

            # Create a DataFrame and filter
            df = pd.DataFrame({"datetime": data_dates, "values": data_values})
            df.set_index("datetime", inplace=True)
            filtered_df = df.loc[start_datetime:end_datetime]

            # Plot using Pandas (example with simple plot)
            st.line_chart(filtered_df)

########################################
# Heatmap and Additional Charts
with tabs[1]:
    st.markdown(f'##### Heatmap chart with hourly data')
    st.write('')

    chart_cols = st.columns(len(epw_names))
    for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
        with col:
            st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_name}</h6>", unsafe_allow_html=True)

            # Extract and filter the data
            data_col = epw_col._get_data_by_field(fields[hourly_selected])
            data_values = data_col.values
            data_dates = data_col.datetimes

            # Get year from the data for dynamic filtering
            data_year = data_dates[0].year if data_dates else 2023
            start_datetime = pd.Timestamp(year=data_year, month=data_start_month, day=data_start_day, hour=data_start_hour)
            end_datetime = pd.Timestamp(year=data_year, month=data_end_month, day=data_end_day, hour=data_end_hour)

            df = pd.DataFrame({"datetime": data_dates, "values": data_values})
            df.set_index("datetime", inplace=True)
            filtered_df = df.loc[start_datetime:end_datetime]

            # Heatmap logic (replace with your heatmap plotting function)
            st.write(filtered_df)  # Placeholder for heatmap generation



########################################
# Hourly Charts - HEATMAP
with tabs[2]:

    st.markdown(f'##### Heatmap chart with hourly data')
    st.write('')

    chart_cols = st.columns(len(epw_names))
    for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
        with col:
            # st.write(epw_col)
            st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_name}</h6>", unsafe_allow_html=True)
            data_col = epw_col._get_data_by_field(fields[hourly_selected])
            data_figure = get_hourly_data_figure(
                data_col, global_colorset, data_conditional_statement,
                data_min, data_max, data_start_month, data_start_day,
                data_start_hour, data_end_month, data_end_day, data_end_hour
            )
            if isinstance(data_figure, str):
                col.error(data_figure)
            else:
                col.plotly_chart(data_figure, use_container_width=True, config=get_figure_config(f'{data_col.header.data_type}'))

