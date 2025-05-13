# IMPORT LIBRARIES
import os
import pathlib
# import datetime
# from datetime import datetime, date

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

    # absrd_tight_hr_spacing()

    ##########
    chart_cols = st.columns(len(epw_names))
    for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
        with col:
            # Extract the data and timestamps
            data_col = epw_col._get_data_by_field(fields[hourly_selected])
            data_values = data_col.values
            data_dates = data_col.datetimes

            # Get year from the data for dynamic filtering
            data_year = data_dates[0].year if data_dates else 2023

    # absrd__horiz_spacing_tight()



    ##########
    st.markdown("#### Filter data by date/time and values")
    d = st.date_input(
        "Select start & end dates",
        (date(data_year, 1, 1), date(data_year, 12, 31)),
        date(data_year, 1, 1),
        date(data_year, 12, 31),
        format="MM.DD.YYYY",
    )
    data_start_month = d[0].month;  data_end_month = d[1].month;  data_start_day = d[0].day;    data_end_day = d[1].day
    sub_col_1, sub_col_2 = st.columns([1, 1])
    data_start_hour = sub_col_1.number_input('Start hour', min_value=0, max_value=23, value=0, key='data_start_hour')
    data_end_hour = sub_col_2.number_input('End hour', min_value=0, max_value=23, value=23, key='data_end_hour')


    # with st.expander(label='Filter data - Values'):
    sub_col_1, sub_col_2 = st.columns([1, 1])
    data_min = sub_col_1.text_input('Min Value')
    data_max = sub_col_2.text_input('Max Value')
    # data_conditional_statement = st.text_input('Apply conditional statement', help=help_text_01)







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



# DEFINE AXIS MIN AND MAX VALUES FOR PLOTTING
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










########################################
# Tabs for Hourly and Daily Charts
st.markdown("""
<style>
	.stTabs [data-baseweb="tab-list"] {
		gap: 5px;
    }
	.stTabs [data-baseweb="tab"] {
		height: 30px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
		border-radius: 5px 5px 0px 0px;
		gap: 10px;
		padding-top: 5px;
		padding-bottom: 5px;
        padding-left: 5px;
        padding-right: 5px;
    }
	.stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
	}
</style>""", unsafe_allow_html=True)



tabs = st.tabs([
    "**HOURLY** Line Chart",
    "**HOURLY** Heatmap",
    "**MONTHLY** Stacked Columns",
    "**MONTHLY** Metrics",
    "SUNPATH",
    "**WINDROSE**",
])







########################################
# Hourly Chart - LINE
with tabs[0]:


    widget_col, spacing, chart_col = st.columns([10,1,80])

    with widget_col:

        # Radio button for chart display mode
        chart_mode = st.radio("Chart Display Mode", ["Side-by-Side", "Combined"], horizontal=True)

    with chart_col:

        if chart_mode == "Side-by-Side":
            # Display charts side by side
            chart_cols = st.columns(len(epw_names))
            for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
                with col:

                    st.write('')
                    custom_filled_region(
                        f"{epw_name}",
                        bg_color="black",
                        text_color="green",
                        border_style="top-bottom",
                        border_color="black", border_thickness="2px",
                        width_percentage=100,
                        )
                    # custom_filled_region(
                    #     f"{epw_name}",
                    #     # bg_color="#F0F2F6",
                    #     text_color="blue",
                    #     border_style="bottom", border_color="grey", border_thickness="1px",
                    #     width_percentage=100,
                    #     )


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


                    # Create Plotly figure
                    fig_line = go.Figure()

                    # Add line trace
                    fig_line.add_trace(go.Scatter(
                        x=filtered_df.index,
                        y=filtered_df["values"],
                        mode="lines",
                        name=epw_name
                    ))

                    # Update layout
                    fig_line.update_layout(
                        yaxis=dict(range=[axis_min, axis_max]),
                        xaxis=dict(
                            rangeselector=dict(
                                buttons=[
                                    dict(count=7, label="1w", step="day", stepmode="backward"),
                                    dict(count=1, label="1m", step="month", stepmode="backward"),
                                    dict(count=3, label="3m", step="month", stepmode="backward"),
                                    dict(step="all")
                                ]
                            ),
                            rangeslider=dict(visible=True)
                        ),
                        template="plotly_white",
                        height=400,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )

                    # Render the chart
                    st.plotly_chart(fig_line, use_container_width=True)

        elif chart_mode == "Combined":

            # Create Plotly figure
            fig_combined = go.Figure()


            for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
                with col:

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


                    # Add trace for each dataset
                    fig_combined.add_trace(go.Scatter(
                        x=filtered_df.index,
                        y=filtered_df["values"],
                        mode="lines",
                        name=epw_name
                    ))

            # Update layout
            fig_combined.update_layout(
                yaxis=dict(range=[axis_min, axis_max]),
                xaxis=dict(
                    rangeselector=dict(
                        buttons=[
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(step="all")
                        ]
                    ),
                    rangeslider=dict(visible=True)
                ),
                template="plotly_white",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            # Render the chart
            st.plotly_chart(fig_combined, use_container_width=True)









########################################
# Hourly Charts - HEATMAP
with tabs[1]:


    # st.markdown(f'##### Heatmap chart with hourly data')
    st.write('')


    chart_cols = st.columns(len(epw_names))
    for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):

        with col: 

            # st.write('')
            custom_filled_region(
                f"{epw_name}",
                bg_color="black",
                text_color="blue",
                border_style="bottom",
                border_color="black", border_thickness="1px",
                width_percentage=80,
                )

            # custom_filled_region(f"{epw_name}", bg_color="#F0F2F6",
            #                         text_color="black", 
            #                         border_style="bottom", border_color="grey", border_thickness="1px",
            #                         width_percentage=70)


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
            

            # Streamlit UI elements
            value_col = 'values'
            agg_func = 'mean'
            # colorscale = st.selectbox("Select Color Scale", ["Viridis", "Plasma", "Cividis", "Blues", "Reds"])

            rgba_colors = get_colors(switch=False, global_colorset=global_colorset)
            # st.write(rgba_colors)

            plotly_colorscale = convert_rgba_to_plotly_colorscale(rgba_colors)

            # Plot heatmap
            fig_heatmap = plot_heatmap_from_datetime_index(
                data=filtered_df, value_col=value_col, agg_func=agg_func, axis_min=axis_min, axis_max=axis_max, custom_colorscale=plotly_colorscale
                )

            st.plotly_chart(fig_heatmap)





########################################











########################################
with tabs[2]:

    # st.markdown('##### Stacked bars with binned data')
    st.write('')

    # User inputs for binning
    widget_col, spacing, chart_col = st.columns([10,1,80])


    with widget_col.container():
        st.caption("Binning Parameters")
        bin_min_val = st.number_input("Min value", min_value=0, value=0 )
        bin_max_val = st.number_input("Max value", max_value=1000, value=40)
        bin_step = st.number_input("Step value", min_value=0, value=5)

        # User input for normalization
        bin_normalize = st.toggle("Show \% of total hours")


    start_month = data_start_month
    end_month = data_end_month


    with chart_col:
        chart_cols = st.columns(len(epw_names))

    for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
        with col:

            st.write('')
            custom_filled_region(f"{epw_name}", bg_color="#F0F2F6",
                                    text_color="black", 
                                    border_style="bottom", border_color="grey", border_thickness="1px",
                                    width_percentage=70)


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

########################################








######################################## METRICS DISPLAY
with tabs[3]:
    # st.markdown('##### Metrics Summary')
    thresholds = {}  # Define a threshold dictionary to store user-defined thresholds for each EPW file
    col_number_input, col_all_metrics = st.columns([10,45])

    with col_number_input:
        col_number_input, spacing = st.columns([10, 1])
        with col_number_input:
            var_name = str(data_col.header.data_type)
            var_unit = str(data_col.header.unit)

            # Create input boxes for three thresholds for the selected variable
            st.write(''); st.write(''); st.write(''); st.write(''); st.write('')
            threshold_01 = st.number_input(
                f"Threshold for {var_name} ({var_unit})", value=28.0, format="%0.1f", step=1.0, key=f"threshold_01"
            )
            custom_hr()
            # absrd_tight_hr_spacing()

            threshold_02 = st.number_input(
                f"Threshold for {var_name} ({var_unit})", value=30.0, format="%0.1f", step=1.0, key=f"threshold_02"
            )
            custom_hr()

            threshold_03 = st.number_input(
                f"Threshold for {var_name} ({var_unit})", value=32.0, format="%0.1f", step=1.0, key=f"threshold_03"
            )




    with col_all_metrics:
        metric_cols = st.columns(len(epw_names))
        chart_cols = st.columns(len(epw_names))
        
        for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
            with col:

    
                custom_filled_region(
                    f"{epw_name}",
                    bg_color="blue",
                    text_color="green",
                    border_style="top-bottom",
                    border_color="black", border_thickness="2px",
                    width_percentage=80,
                    )

                st.write(''); st.write('')

                # Extract and process data from EPW collection
                data_col = epw_col._get_data_by_field(fields[hourly_selected])

                # Apply the analysis period to the data
                plot_analysis_period = [data_start_month, data_start_day, data_start_hour, data_end_month, data_end_day, data_end_hour]
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




with tabs[4]:

    widget_col, spacing, chart_col = st.columns([10,1,80])
    with chart_col:
        chart_cols = st.columns(len(epw_names))

    for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):

        with col:

            st.write('')
            custom_filled_region(f"{epw_name}", bg_color="#F0F2F6",
                                    text_color="black", 
                                    border_style="bottom", border_color="grey", border_thickness="1px",
                                    width_percentage=70)

            
            var_name = str(data_col.header.data_type)
            var_unit = str(data_col.header.unit)

            st.write('')
            st.write('')
            # st.markdown(f'##### Sunpath & {var_name}')


            # Extract the data and timestamps
            data_col = epw_col._get_data_by_field(fields[hourly_selected])
            data_values = data_col.values
            data_dates = data_col.datetimes

            # Apply the analysis period to the data
            plot_analysis_period = [data_start_month, data_start_day, data_start_hour, data_end_month, data_end_day, data_end_hour]
            plot_data = absrd_apply_analysis_period(data_col, plot_analysis_period)

            # Convert data_col.values to a list of numeric values if they are stored as tuples
            numeric_values = [val[0] if isinstance(val, tuple) else val for val in data_col.values]


            sunpath_figure = get_sunpath_figure(
                'with epw data', global_colorset, epw=epw_col, switch=None, data=data_col
            )
            
            st.plotly_chart(sunpath_figure, use_container_width=True,
                            config=get_figure_config(f'Sunpath_{epw_name}'))












with tabs[5]:
    widget_col, spacing, chart_col = st.columns([10, 2, 80])


    with widget_col.container():
        st.markdown("Wind Rose Parameters")
        bin_windrose_min_val = st.number_input("Min Wind speed", min_value=0, value=0 )
        bin_windrose_max_val = st.number_input("Max Wind speed", max_value=100, value=10)
        bin_windrose_step = st.number_input("Step for Wind speed", min_value=0, value=2)
        windrose_sectors = st.radio("Number of sectors", options=[4, 8, 16, 32], index=2, horizontal=True)



    with chart_col:
        chart_cols = st.columns(len(epw_names))

    for col, epw_col, epw_name in zip(chart_cols, epw_files, epw_names):
        with col:

            st.write('')
            custom_filled_region(
                f"{epw_name}",
                bg_color="#F0F2F6",
                text_color="black",
                border_style="bottom",
                border_color="grey",
                border_thickness="1px",
                width_percentage=70,
            )

            # --- Extract wind speed and direction ---
            wind_speed_data = epw_col.wind_speed
            wind_dir_data = epw_col.wind_direction

            # --- Apply analysis period ---
            plot_analysis_period = [data_start_month, data_start_day, data_start_hour,
                                    data_end_month, data_end_day, data_end_hour]

            wind_speed_filtered = absrd_apply_analysis_period(wind_speed_data, plot_analysis_period)
            wind_dir_filtered = absrd_apply_analysis_period(wind_dir_data, plot_analysis_period)

            # --- Convert values to numeric lists ---
            wind_speed = [val[0] if isinstance(val, tuple) else val for val in wind_speed_filtered.values]
            wind_dir = [val[0] if isinstance(val, tuple) else val for val in wind_dir_filtered.values]


            wind_speed_bins = np.arange(bin_windrose_min_val,bin_windrose_max_val+1,bin_windrose_step)


            # --- Get color scale from global_colorset ---
            rgba_colors = get_colors(switch=False, global_colorset=global_colorset)
            fig = plot_windrose(
                wind_direction=wind_dir, wind_speed=wind_speed,
                num_sectors=windrose_sectors,
                speed_bins=wind_speed_bins,
                unit='m/s',
                color_map=rgba_colors,
            )
            st.plotly_chart(fig, use_container_width=True)
