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
if len(epw_names)<1:    
    st.warning("EPW files are not uploaded. Please go to the 'Data Upload' page to upload your EPW files.")
    st.stop()



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


    # ##########
    # # User inputs for binning
    # st.markdown("#### Stacked Columns - Binning Parameters")
    # with st.expander(label='Binning Parameters - Monthly data'):
    #     sub_col_1, sub_col_2, sub_col_3 = st.columns([1,1,1])
    #     bin_min_val = sub_col_1.number_input("Min value", min_value=0, value=0 )
    #     bin_max_val = sub_col_2.number_input("Max value", max_value=50, value=40)
    #     bin_step = sub_col_3.number_input("Step value", min_value=1, value=5)

    #     # User input for normalization
    #     # bin_normalize = st.sidebar.radio("Display Mode", ["Total Hours", "% Hours"])
    #     bin_normalize = st.toggle("Show \% of total hours")

    # custom_hr()

    plot_analysis_period = [data_start_month, data_start_day, data_start_hour, data_end_month, data_end_day, data_end_hour]




####################################################################################
all_values = []
chart_cols = st.columns(len(epw_names))
for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
    with col:
        lb_ap = AnalysisPeriod(
            data_start_month, data_start_day, data_start_hour,
            data_end_month, data_end_day, data_end_hour,
            )
        col = col.filter_by_analysis_period(lb_ap)
        data_values = col.values  # Extracts the actual data values (temperature, radiation, etc.)
        data_dates = col.datetimes  # Extracts corresponding datetime objects for each hourly reading

        all_values.append( max(data_values) )
        all_values.append( min(data_values) )

min_value = min(all_values)
max_value = max(all_values)
axis_min, axis_max = define_yaxis_range(var_min=min_value, var_max=max_value)
axis_min = axis_min*1.05; axis_max = axis_max*1.05






####################################################################################
# Tabs for Hourly and Daily Charts
tabs = st.tabs([
    "Hourly - **Line Chart with Ladybug**",
    "Hourly - **Line Chart with Plotly**",
    "Hourly - Ladybug - **Heatmap**",
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
            # st.write(epw_col)
            st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_name}</h6>", unsafe_allow_html=True)
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
# Hourly Chart - LINE
with tabs[1]:

    st.markdown(f'##### Line chart with hourly data')
    st.write('')

    all_values = []
    chart_cols = st.columns(len(epw_names))
    for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
        with col:
            lb_ap = AnalysisPeriod(
                data_start_month, data_start_day, data_start_hour,
                data_end_month, data_end_day, data_end_hour,
                )
            data_col = data_col.filter_by_analysis_period(lb_ap)
            data_values = data_col.values  # Extracts the actual data values (temperature, radiation, etc.)
            data_dates = data_col.datetimes  # Extracts corresponding datetime objects for each hourly reading

            all_values.append( max(data_values) )
            all_values.append( min(data_values) )

    min_value = min(all_values)
    max_value = max(all_values)
    axis_min, axis_max = define_yaxis_range(var_min=min_value, var_max=max_value)
    axis_min = axis_min*1.05; axis_max = axis_max*1.05


    for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
        with col:
            # st.write(epw_col)
            st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_name}</h6>", unsafe_allow_html=True)
            data_col = epw_col._get_data_by_field(fields[hourly_selected])


            lb_ap = AnalysisPeriod(
                data_start_month, data_start_day, data_start_hour,
                data_end_month, data_end_day, data_end_hour,
                )
            data_col = data_col.filter_by_analysis_period(lb_ap)

            # Assuming `data_col` is the result of epw_col._get_data_by_field(fields[hourly_selected])
            data_values = data_col.values  # Extracts the actual data values (temperature, radiation, etc.)
            data_dates = data_col.datetimes  # Extracts corresponding datetime objects for each hourly reading

            var_name = str(data_col.header.data_type)
            var_unit = str(data_col.header.unit)

            # Convert to DataFrame
            df = pd.DataFrame({"datetime": data_dates, "values": data_values})
            df.set_index("datetime", inplace=True)  # Set Date as the index


            # fig = absrd__line_chart_daily__range(hourly_data=df)
            fig = absrd__line_chart_daily__range(
                hourly_data=df,
                show_legend=True, legend_position="bottom left",
                margins=(40, 10, 10, 0),        # Margins for the layout in the format (top, right, bottom, left).
                fixed_y_range=(axis_min, axis_max),
                var_name=var_name, var_unit=var_unit,
                )

            # Display chart in Streamlit
            st.plotly_chart(fig)







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





########################################
with tabs[3]:
    # Create sub-columns for selection widgets
    sub_col_11, spacing, sub_col_12, spacing = st.columns([6, 1, 3, 1])

    # Multi-select box for selecting EPW files to plot
    with sub_col_11:
        selected_files = st.multiselect("Select files to plot", options=epw_names, default=epw_names)

    # Radio button to choose layout (horizontal or vertical)
    with sub_col_12:
        layout = st.radio("Select plotting layout", options=["Horizontal", "Vertical"], horizontal=True)


    st.markdown('##### Scatter chart with hourly data & Line chart with average day')

    # Display charts based on selected layout
    if layout == "Horizontal":
        # Create one row with multiple columns for each selected file
        chart_cols = st.columns(len(selected_files))
        
        # Loop through each selected file and display the chart in its respective column
        for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
            if epw_name in selected_files:
                with col:
                    # Add the EPW file name as a header
                    st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_name}</h6>", unsafe_allow_html=True)
                    
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
        for epw_col, epw_name in zip(epw_files, epw_names):
            if epw_name in selected_files:
                # Add a separator and the EPW file name as a header
                st.markdown("---")
                st.markdown(f"<h6 style='text-align: center; color: black;'>{epw_name}</h6>", unsafe_allow_html=True)
                
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



