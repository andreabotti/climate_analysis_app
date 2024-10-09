# IMPORT LIBRARIES
import os, requests, pathlib, json
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
epw_names = [item['epw_file_name'] for item in epw_files_list]
epw_files = [item['epw_object'] for item in epw_files_list]
st.session_state['epw_names'] = epw_names
st.session_state['epw_files'] = epw_files

# Function to check if required keys exist in session state
if len(epw_names) < 1:    
    st.warning("EPW files are not uploaded. Please go to the 'Data Upload' page to upload your EPW files.")
    st.stop()


####################################################################################
# Sidebar Input Parameters
st.sidebar.header("Input Parameters")

col_sidebar_1, col_sidebar_2 = st.sidebar.columns([2, 2])

start_year = col_sidebar_1.number_input(
    label="Start Year",
    min_value=1985,
    max_value=2100,
    value=2023,
    step=1
)
end_year = col_sidebar_2.number_input(
    label="End Year",
    min_value=1985,
    max_value=2100,
    value=2023,
    step=1
)

tilt_angle = col_sidebar_1.slider(
    label="Tilt Angle (degrees)",
    min_value=0,
    max_value=90,
    value=35,
    step=1
)
azimuth = col_sidebar_2.slider(
    label="Azimuth (degrees)",
    min_value=0,
    max_value=360,
    value=180,
    step=1
)

peak_power = col_sidebar_1.number_input(
    label="Peak Power (kWp)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1
)
system_loss = col_sidebar_2.slider(
    label="System Loss (%)",
    min_value=0,
    max_value=30,
    value=14,
    step=1
)

# Validate years
if end_year < start_year:
    st.sidebar.error("End Year must be greater than or equal to Start Year.")




####################################################################################
# Button to trigger API call
if st.sidebar.button("Fetch PVGIS Data"):

    if end_year < start_year:
        st.error("Please ensure that End Year is not earlier than Start Year.")
    else:
        with st.spinner('Fetching data from PVGIS API...'):
            # Store data for each EPW file
            hourly_data_dict = {}
            pv_production_data_dict = {}
            df_hourly_dict = {}
            df_pv_monthly_dict = {}
            df_pv_totals_dict = {}
            bullet_points_dict = {}

            # Fetch data for each EPW file
            for epw_name, epw_file in zip(epw_names, epw_files):
                # Fetch hourly data
                hourly_data = fetch_pvgis_hourly_data(
                    lat=epw_file.location.latitude,
                    lon=epw_file.location.longitude,
                    startyear=start_year,
                    endyear=end_year,
                    peakpower=peak_power,
                    loss=system_loss,
                    angle=tilt_angle,
                    azimuth=azimuth
                )

                # Fetch PV production data
                pv_production_data = fetch_pv_production_data(
                    lat=epw_file.location.latitude,
                    lon=epw_file.location.longitude,
                    peakpower=peak_power,
                    loss=system_loss,
                    tilt=tilt_angle,
                    azimuth=azimuth,
                )

                hourly_data_dict[epw_file] = hourly_data
                pv_production_data_dict[epw_file] = pv_production_data

                # Convert 'outputs' to DataFrame for hourly data
                outputs = hourly_data.get('outputs', {}) if hourly_data else {}
                df_hourly = convert_outputs_to_dataframe(outputs)
                df_hourly_dict[epw_file] = df_hourly

                # Parse PV production data into DataFrames
                if pv_production_data:
                    df_pv_monthly, df_pv_totals = parse_pv_production_data(pv_production_data)
                    bullet_points = extract_meta_monthly_variable_info(pv_production_data)
                else:
                    df_pv_monthly, df_pv_totals, bullet_points = pd.DataFrame(), pd.DataFrame(), []

                df_pv_monthly_dict[epw_file] = df_pv_monthly
                df_pv_totals_dict[epw_file] = df_pv_totals
                bullet_points_dict[epw_file] = bullet_points

        ####################################################################################
        # Create three tabs: Charts, Tables and Bullets, and Raw Output
        tabs = st.tabs(["📊 Charts", "📋 Tables and Bullets", "📄 Raw Output"])

        # Tab for visualizing the charts
        with tabs[0]:
            st.markdown(f'##### PVGIS Data Visualization - Charts')
            chart_cols = st.columns(len(epw_names))
            for col, epw_name, epw_file in zip(chart_cols, epw_names, epw_files):
                with col:

                    lat = round(epw_file.location.latitude,3); lon = round(epw_file.location.longitude,3)
                    station = epw_file._location.city
                    st.markdown(
                        f"<h5 style='text-align: center; color: black;'>{station}  lat: {lat} - lon: {lon}</h5>",
                        unsafe_allow_html=True,
                        )

                    # Define custom y-axis ranges for both subplots
                    range_min_em, range_max_em = 0, 265
                    range_min_him, range_max_him = 0, 265

                    # Define custom margins
                    custom_margins = dict(l=20, r=20, t=60, b=20)

                    # Plot the subplots
                    fig_plotly = plot_pv_monthly_comparison_subplots(
                        df_pv_monthly_dict[epw_file], tilt_angle, azimuth, 
                        range_min_em, range_max_em, range_min_him, range_max_him, margin=custom_margins
                    )
                    st.plotly_chart(fig_plotly, use_container_width=True)

        # Tab for displaying tables and bullet points
        with tabs[1]:
            st.markdown(f'##### PVGIS Data - Tables and Bullets')
            chart_cols = st.columns(len(epw_files))  # Create columns for each EPW file
            for col, epw_file in zip(chart_cols, epw_files):
                with col:
                    
                    # Display Monthly DataFrame
                    if not df_pv_monthly_dict[epw_file].empty:
                        st.write(f"Monthly PV Production Data for {epw_file}:")
                        table = df_pv_monthly_dict[epw_file].set_index('month')
                        st.dataframe(table, width=500, height=460)

                    # Display Totals DataFrame
                    if not df_pv_totals_dict[epw_file].empty:
                        st.write(f"Yearly PV Production Totals for {epw_file}:")
                        st.dataframe(df_pv_totals_dict[epw_file])

                    # Display Bullet Points
                    custom_hr()
                    st.write(f"**List of Variables**")
                    for bullet in bullet_points_dict[epw_file]:
                        st.caption(bullet)

        # Tab for displaying raw JSON output
        with tabs[2]:
            st.markdown(f'##### PVGIS Data - Raw Output')
            chart_cols = st.columns(len(epw_files))  # Create columns for each EPW file
            for col, epw_file in zip(chart_cols, epw_files):
                with col:
                    # st.markdown(f"### {epw_file}")
                    with st.expander(f"Hourly Data (Raw JSON) for {epw_file}"):
                        st.json(hourly_data_dict[epw_file])

                    with st.expander(f"PV Production Data (Raw JSON) for {epw_file}"):
                        st.json(pv_production_data_dict[epw_file])




####################################################################################
# Button to trigger API call

azimuth_values = range(-180, 181, 10)  # Azimuth from -180 to 180 with steps of 10
tilt_values = range(0, 91, 5)          # Tilt angle from 0 to 90 with steps of 5

azimuth_values = range(-180, 181, 30)  # Azimuth from -180 to 180 with steps of 10
tilt_values = range(0, 91, 10)          # Tilt angle from 0 to 90 with steps of 5


with st.sidebar:
    custom_hr()

if st.sidebar.button("Fetch PVGIS Data for multiple values of tilt and azimuth (calc. time > 2 mins)"):

    with st.spinner('Fetching data from PVGIS API...'):

        chart_cols = st.columns(len(epw_names))
        for col, epw_name, epw_file in zip(chart_cols, epw_names, epw_files):
            with col:

                lat = round(epw_file.location.latitude,3); lon = round(epw_file.location.longitude,3)
                station = epw_file._location.city
                st.markdown(
                    f"<h5 style='text-align: center; color: black;'>{station}  lat: {lat} - lon: {lon}</h5>",
                    unsafe_allow_html=True,
                    )

                # Run the iteration function and retrieve results
                data = iterate_pv_production(
                    lat= epw_file.location.latitude, lon=epw_file.location.longitude,
                    peakpower=peak_power, loss=system_loss,
                    azimuth_values=azimuth_values, tilt_values=tilt_values,
                    pause_duration=0.2,
                    )

                if data:
                    df = pd.DataFrame(data)
                    # col.write(df)                

                output_cols = df.columns.drop( ['azimuth', 'tilt','daily_energy', 'daily_irr', 'total_loss'] )
                # st.write(output_cols)

                for oc in output_cols:
                    df_filtered = df[['azimuth', 'tilt', oc]]

                    # Pivot the DataFrame to create a matrix
                    energy_matrix = df_filtered.pivot(index='tilt', columns='azimuth', values=oc).round(0)

                    # Define the dictionary with the required keys and values
                    metrics_dict = {
                        'monthly_energy': {'metric': '[kWh/mo]', 'range': [0, 140], 'colormap': 'BuGn'},
                        'yearly_energy': {'metric': '[kWh/yr]', 'range': [0, 1500], 'colormap': 'BuGn'},
                        'monthly_irr': {'metric': '[kWh/m²/mo]', 'range': [0, 200], 'colormap': 'Oranges'},
                        'yearly_irr': {'metric': '[kWh/m²/yr]', 'range': [0, 2000], 'colormap': 'Oranges'}
                    }

                    # Function to return details based on the output_var
                    def get_metric_details(output_var):
                        # Check if the output_var exists in the dictionary
                        if output_var in metrics_dict:
                            return metrics_dict[output_var]
                        else:
                            return f"'{output_var}' is not a valid key. Please choose a valid output_var."
                    
                    var_metric = get_metric_details(oc)['metric']

                    st.write('')
                    st.markdown(f'##### Output values for {oc} - {var_metric}')

                    # Display the transformed matrix
                    if oc == 'monthly_energy':
                        energy_matrix = energy_matrix.applymap(lambda x: f"{x:.0f}")
                        energy_matrix__styled = absrd_style_df_streamlit(df=energy_matrix, range_min_max=[0,140], colormap='BuGn')
                        st.dataframe(energy_matrix__styled, use_container_width=True)

                    if oc == 'yearly_energy':
                        energy_matrix = energy_matrix.applymap(lambda x: f"{x:.0f}")
                        energy_matrix__styled = absrd_style_df_streamlit(df=energy_matrix, range_min_max=[0,1500], colormap='BuGn')
                        st.dataframe(energy_matrix__styled, use_container_width=True)

                    if oc == 'monthly_irr':
                        energy_matrix = energy_matrix.applymap(lambda x: f"{x:.0f}")
                        energy_matrix__styled = absrd_style_df_streamlit(df=energy_matrix, range_min_max=[0,200], colormap='Oranges')
                        st.dataframe(energy_matrix__styled, use_container_width=True)

                    if oc == 'yearly_irr':
                        energy_matrix = energy_matrix.applymap(lambda x: f"{x:.0f}")
                        energy_matrix__styled = absrd_style_df_streamlit(df=energy_matrix, range_min_max=[0,2000], colormap='Oranges')
                        st.dataframe(energy_matrix__styled, use_container_width=True)


                custom_hr()
                
                with st.expander('Raw JSON data'):
                    st.write(data)
                    
