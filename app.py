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

st.markdown('Use the navigation links below to go to different sections of the application.')

custom_hr()

# Sidebar or Main Page Navigation
st.markdown("##### Navigation")
st.markdown("""
- [Data Upload](data_upload)
- [Hourly Diagrams](hourly_diagrams)
- [Monthly Diagrams](monthly_diagrams)
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
