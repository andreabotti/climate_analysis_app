# IMPORT LIBRARIES
import os, re
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
stat_names = [item['stat_file_name'] for item in epw_files_list]
stat_file_paths = [item['stat_file_path'] for item in epw_files_list]

st.session_state['epw_names'] = epw_names
st.session_state['epw_files'] = epw_files

# Function to check if required keys exist in session state
if len(epw_names)<1:    
    st.warning("EPW files are not uploaded. Please go to the 'Data Upload' page to upload your EPW files.")
    st.stop()

LOCAL_ROOT_PATH = './data/'




# Define the row slicing ranges
slicing_ranges = [
    (0, 4),   # Rows 1-4 (0-based index)
    (4, 6),   # Rows 5-6
    (6, 9),   # Rows 7-9
    (9, 12)   # Rows 10-12
]
dict_titles = {
    0 : 'Max and Min Temperatures',
    1 : 'Daily Avg Temperatures',
    2 : 'DayTime Max and Min Temperatures',
    3 : 'NightTime Max and Min Temperatures',
}




chart_cols = st.columns(len(stat_names))
for col, stat_name, epw_file in zip(chart_cols, stat_names, epw_files):
    with col:
        # st.write(epw_col)
        st.markdown(f"<h5 style='text-align: center; color: black;'>{stat_name}</h5>", unsafe_allow_html=True)
        
        stat_file_path = next((item["stat_file_path"] for item in epw_files_list if item["stat_file_name"] == stat_name), None)
        # st.caption(stat_file_path)
        # custom_hr()

        dry_bulb_df_final, summary_dict = absrd_process_stat_file(file_path=stat_file_path)

        # Create two columns
        col1, col2 = st.columns(2)

        # Display metrics in the columns
        for idx, (key, value) in enumerate(summary_dict.items()):
            label = key.replace("Dry Bulb Temperature", "DBT")
            label = key
            help_info = "Dry Bulb Temperature"

            # Place first metric in the first column, second in the second column
            if idx % 2 == 0:
                col1.metric(label=label, value=value)
            else:
                col2.metric(label=label, value=value)





        # Loop through the slicing ranges and create sub-DataFrames
        i=0
        for start, end in slicing_ranges:

            custom_hr()
            st.markdown(f'##### {dict_titles[i]}')

            # Slice the DataFrame based on the index range
            sub_df = dry_bulb_df_final.iloc[start:end]

            if i>0:
                sub_df__styled = absrd_style_df_streamlit(df=sub_df, range_min_max=[-10,40], colormap='RdBu_r')
                st.dataframe(sub_df__styled, use_container_width=True)
                if i==2:
                    df_daytime = sub_df
                if i==3:
                    df_nighttime = sub_df

            else:
                # Reset the index to convert the index into a regular column
                df_reset = sub_df.reset_index()

                # Apply the custom row styling function using .apply with axis=1 for rows
                styled_df = df_reset.style.apply(lambda x: row_style(df_reset,x.name), axis=1)            
                st.dataframe(styled_df, use_container_width=True)

            i+=1


        custom_hr()
        st.markdown(f'##### Daytime and NightTime Temperatures')

        fig_daytime_nighttime = absrd_daytime_nighttime_scatter(df_daytime, df_nighttime)
        st.plotly_chart(fig_daytime_nighttime)