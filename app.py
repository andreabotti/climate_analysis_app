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
# INTRODUCTION
st.markdown('##### Welcome to the Climate Analysis App.')

st.markdown('This app allows to visualise variables from weather data in EnergyPlus Weather Format (EPW). \
            For more information visit [this link](https://designbuilder.co.uk/cahelp/Content/EnergyPlusWeatherFileFormat.htm) \
            or [this link](https://climate.onebuilding.org/papers/EnergyPlus_Weather_File_Format.pdf)')



custom_hr()

# Sidebar or Main Page Navigation
st.markdown("##### Navigation")
st.markdown('Please navigate the following pages to go to different sections of the application.')
st.markdown("""
- _1_Upload_or_Select_EPW_Data_: use this page to upload EPW and STAT files, or to choose from the app archive (currently coverint TMYs files for Italy) 
- _2_View_Summary_Statistics_: visit this page to see statistics including min and max yearly, monthly and daily temperatures
- _3_View_Hourly_Charts_: visit this page to plot scatter and heatmap charts
- _4_PV_ProductionEstimate_: visit this page to produce PV production estimates using the PVGIS API
""", unsafe_allow_html=True)

custom_hr()
st.markdown("""
You can find a wide range of EPW weather data by visiting:
- https://www.ladybug.tools/epwmap/
- https://climate.onebuilding.org/            
""")

####################################################################################
# Step 1: Inject the CSS for the Google Font you want to use
def load_google_fonts(font_name: str):
    """
    Load Google Font into Streamlit application.
    Args:
        font_name (str): Name of the Google Font to be loaded.
    """
    font_url = f"https://fonts.googleapis.com/css2?family={font_name.replace(' ', '+')}:wght@400;600;700&display=swap"
    st.markdown(
        f"""
        <style>
        @import url('{font_url}');

        /* Apply the custom font to body and all elements */
        * {{
            font-family: '{font_name}', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Example usage of loading Google Font "Roboto"
# load_google_fonts("Roboto")




# Dictionary mapping short codes to full region names
ITA_regions_dict = {
    "AB": "Abruzzo",
    "BC": "Basilicata",
    "CM": "Campania",
    "ER": "Emilia-Romagna",
    "FV": "Friuli-Venezia Giulia",
    "LB": "Lombardia",
    "LG": "Liguria",
    "LM": "Lazio",
    "LZ": "Lazio",
    "MH": "Marche",
    "ML": "Molise",
    "PM": "Piemonte",
    "PU": "Puglia",
    "SC": "Sicilia",
    "SD": "Sardegna",
    "TC": "Toscana",
    "TT": "Trentino-Alto Adige",
    "UM": "Umbria",
    "VD": "Valle d'Aosta",
    "VN": "Veneto"
}

# Create a reverse mapping for easy lookup
full_to_short = {full: short for short, full in ITA_regions_dict.items()}

# Streamlit selectbox for choosing full region names
# selected_region = st.selectbox("Choose an Italian region:", list(full_to_short.keys()))

# Display the short code for the selected region
# st.write(f"Short code for {selected_region}: {full_to_short[selected_region]}")


# Define the file path
csv_file_path = "./data/Europe_ITA_TMYx_locations.csv"

# Step 1: Read the Excel file and load it into a DataFrame
df = pd.read_csv(
    csv_file_path,
    index_col=0)

df.drop(['Time Zone (GMT +/-)'], axis=1, inplace=True)


# st.dataframe(df, use_container_width=True)



st.session_state['ITA_regions_dict'] = ITA_regions_dict
st.session_state['df_locations_ITA'] = df


