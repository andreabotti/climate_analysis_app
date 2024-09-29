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


# Example usage of loading Google Font "Roboto"
# load_google_fonts("Roboto")




# Function to check if required keys exist in session state
def check_epw_files_in_session():
    if 'epw_files' not in st.session_state or 'epw_names' not in st.session_state:
        st.warning("EPW files are not uploaded. Please go to the 'Data Upload' page to upload your EPW files.")
        st.stop()

# Call this function at the beginning of the page
check_epw_files_in_session()



####################################################################################
# LOAD DATA INTO SESSION STATE
epw_files = st.session_state['epw_files'] 
epw_names = st.session_state['epw_names']



####################################################################################
with st.sidebar:

    # Introduced checkboxes to plot charts
    fields = get_fields()
    # st.write(fields)
    hourly_selected = st.selectbox('Select an environmental variable', options=fields.keys(), key='data')
    # custom_hr()
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


    ##########
    # User inputs for binning
    st.markdown("#### Stacked Columns - Binning Parameters")
    with st.expander(label='Binning Parameters - Monthly data'):
        sub_col_1, sub_col_2, sub_col_3 = st.columns([1,1,1])
        bin_min_val = sub_col_1.number_input("Min value", min_value=0, value=0 )
        bin_max_val = sub_col_2.number_input("Max value", max_value=50, value=40)
        bin_step = sub_col_3.number_input("Step value", min_value=1, value=5)

        # User input for normalization
        # bin_normalize = st.sidebar.radio("Display Mode", ["Total Hours", "% Hours"])
        bin_normalize = st.toggle("Show \% of total hours")


    custom_hr()

    plot_analysis_period = [data_start_month, data_start_day, data_start_hour, data_end_month, data_end_day, data_end_hour]






####################################################################################
# Tabs for Hourly and Daily Charts
tabs = st.tabs([
    "Hourly - **Line Chart**",
    "Hourly - **Heatmap**",
    "Hourly - **Avg. Day**",
    "Monthly - **Stacked Columns**",
    "Monthly - **METRICS**",
    ])



########################################
# Hourly Chart - LINE
with tabs[0]:

    st.markdown(f'##### Line chart with hourly data')
    st.write('')

    chart_cols = st.columns(len(epw_names))
    for col, epw_col, epw_file in zip(chart_cols, epw_names, epw_files):
        with col:
            # st.write(epw_col)
            st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_file}</h6>", unsafe_allow_html=True)
            data_col = epw_col._get_data_by_field(fields[hourly_selected])

            data_figure = get_hourly_line_chart_figure(
                data=data_col,
                switch=True,
                global_colorset=global_colorset,
                )

            if isinstance(data_figure, str):
                col.error(data_figure)
            else:
                col.plotly_chart(data_figure, use_container_width=True, config=get_figure_config(f'{data_col.header.data_type}'))




########################################
# Hourly Charts - HEATMAP
with tabs[1]:

    st.markdown(f'##### Heatmap chart with hourly data')
    st.write('')

    chart_cols = st.columns(len(epw_names))
    for col, epw_col, epw_file in zip(chart_cols, epw_names, epw_files):
        with col:
            # st.write(epw_col)
            st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_file}</h6>", unsafe_allow_html=True)
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





########################################
with tabs[2]:
    # Create sub-columns for selection widgets
    sub_col_11, spacing, sub_col_12, spacing = st.columns([6, 1, 3, 1])

    # Multi-select box for selecting EPW files to plot
    with sub_col_11:
        selected_files = st.multiselect("Select files to plot", options=epw_files, default=epw_files)

    # Radio button to choose layout (horizontal or vertical)
    with sub_col_12:
        layout = st.radio("Select plotting layout", options=["Horizontal", "Vertical"], horizontal=True)


    st.markdown('##### Scatter chart with hourly data & Line chart with average day')

    # Display charts based on selected layout
    if layout == "Horizontal":
        # Create one row with multiple columns for each selected file
        chart_cols = st.columns(len(selected_files))
        
        # Loop through each selected file and display the chart in its respective column
        for col, epw_col, epw_file in zip(chart_cols, epw_names, epw_files):
            if epw_file in selected_files:
                with col:
                    # Add the EPW file name as a header
                    st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_file}</h6>", unsafe_allow_html=True)
                    
                    # Extract and process data from EPW collection
                    data_col = epw_col._get_data_by_field(fields[hourly_selected])
                    
                    # Apply the analysis period to the data
                    plot_data = absrd_apply_analysis_period(data_col, plot_analysis_period)
                    
                    # Create the daily profile figure
                    fig_daily = absrd_avg_daily_profile(
                        plot_data=data_col, global_colorset=global_colorset, plot_analysis_period=plot_analysis_period
                    )
                    
                    # Display the daily profile figure for the current EPW file
                    st.plotly_chart(fig_daily)

    else:  # Vertical layout
        # Loop through each selected file and display the chart vertically
        for epw_col, epw_file in zip(epw_names, epw_files):
            if epw_file in selected_files:
                # Add a separator and the EPW file name as a header
                st.markdown("---")
                st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_file}</h6>", unsafe_allow_html=True)
                
                # Extract and process data from EPW collection
                data_col = epw_col._get_data_by_field(fields[hourly_selected])
                
                # Apply the analysis period to the data
                plot_data = absrd_apply_analysis_period(data_col, plot_analysis_period)
                
                # Create the daily profile figure
                fig_daily = absrd_avg_daily_profile(
                    plot_data=data_col, global_colorset=global_colorset, plot_analysis_period=plot_analysis_period
                )
                
                # Display the daily profile figure for the current EPW file
                st.plotly_chart(fig_daily)






########################################
with tabs[3]:

    st.markdown('##### Stacked bars with binned data')
    st.write('')

    start_month = data_start_month
    end_month = data_end_month

    chart_cols = st.columns(len(epw_names))
    for col, epw_col, epw_file in zip(chart_cols, epw_names, epw_files):
        with col:
            # st.write(epw_col)
            st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_file}</h6>", unsafe_allow_html=True)

            data_col = epw_col._get_data_by_field(fields[hourly_selected])

            if start_month > end_month:
                st.sidebar.error("Start month must be less than or equal to End month")
            else:
                df_plot = slice_data_by_month(plot_data = data_col, start_month=start_month, end_month=end_month)


        if bin_min_val >= bin_max_val:
            st.sidebar.error("Min value must be less than Max value")
        else:
            # Process and bin the timeseries data
            binned_data = bin_timeseries_data(df_plot, bin_min_val, bin_max_val, bin_step)

            # if bin_normalize == "% Hours":
            binned_data = normalize_data(binned_data)
            labels = binned_data['binned'].unique()
            
            # Create and display the stacked bar chart
            fig = create_stacked_bar_chart(binned_data=binned_data, color_map=global_colorset, normalize = bin_normalize)
            col.plotly_chart(fig)





######################################## METRICS DISPLAY
with tabs[4]:
    st.markdown('##### Metrics Summary')
    thresholds = {}  # Define a threshold dictionary to store user-defined thresholds for each EPW file
    col_number_input, col_all_metrics = st.columns([1, 3])

    with col_number_input:
        col_number_input, spacing = st.columns([4, 1])
        with col_number_input:
            var_name = str(data_col.header.data_type)
            var_unit = str(data_col.header.unit)

            # Create input boxes for three thresholds for the selected variable
            st.write('')
            st.write('')
            st.write('')
            threshold_01 = st.number_input(
                f"Threshold for {var_name} ({var_unit})", value=28.0, format="%0.1f", step=1.0, key=f"threshold_01"
            )
            custom_hr()

            threshold_02 = st.number_input(
                f"Threshold for {var_name} ({var_unit})", value=30.0, format="%0.1f", step=1.0, key=f"threshold_02"
            )
            custom_hr()

            threshold_03 = st.number_input(
                f"Threshold for {var_name} ({var_unit})", value=32.0, format="%0.1f", step=1.0, key=f"threshold_03"
            )

    with col_all_metrics:
        metric_cols = st.columns(len(epw_names))
        for col, epw_col, epw_file in zip(metric_cols, epw_names, epw_files):
            with col:
                st.markdown(f"**{epw_file}**")
                data_col = epw_col._get_data_by_field(fields[hourly_selected])

            if epw_file in selected_files:
                # Extract and process data from EPW collection
                data_col = epw_col._get_data_by_field(fields[hourly_selected])

                # Apply the analysis period to the data
                plot_data = absrd_apply_analysis_period(data_col, plot_analysis_period)

                # Convert data_col.values to a list of numeric values if they are stored as tuples
                numeric_values = [val[0] if isinstance(val, tuple) else val for val in data_col.values]

                # Calculate the number of values above each user-defined threshold
                data_above_threshold_01 = sum(val > threshold_01 for val in numeric_values)
                data_above_threshold_02 = sum(val > threshold_02 for val in numeric_values)
                data_above_threshold_03 = sum(val > threshold_03 for val in numeric_values)

                total_data_points = len(numeric_values)
                percentage_above_threshold_01 = (data_above_threshold_01 / total_data_points) * 100
                percentage_above_threshold_02 = (data_above_threshold_02 / total_data_points) * 100
                percentage_above_threshold_03 = (data_above_threshold_03 / total_data_points) * 100

                # Display metrics for values above each threshold
                col.metric(
                    label=f"Hours Above {threshold_01}{var_unit}  (% total hours)",
                    value=f"{data_above_threshold_01} ({percentage_above_threshold_01:.1f}%)",
                    delta=None,
                )
                with col:
                    custom_hr()
                col.metric(
                    label=f"Hours Above {threshold_02}{var_unit}",
                    value=f"{data_above_threshold_02} ({percentage_above_threshold_02:.1f}%)",
                    delta=None,
                )
                with col:
                    custom_hr()
                col.metric(
                    label=f"Hours Above {threshold_03}{var_unit}",
                    value=f"{data_above_threshold_03} ({percentage_above_threshold_03:.1f}%)",
                    delta=None,
                )
