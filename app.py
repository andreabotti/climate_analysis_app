# IMPORT LIBRARIES
import os
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
from ladybug.epw import EPW
from fn__libraries import *
# from template_graphs import *

mapbox_access_token = 'pk.eyJ1IjoiYW5kcmVhYm90dGkiLCJhIjoiY2xuNDdybms2MHBvMjJqbm95aDdlZ2owcyJ9.-fs8J1enU5kC3L4mAJ5ToQ'



####################################################################################
# PAGE HEADER
from fn__page_header import create_page_header
create_page_header()



####################################################################################
# LOAD DATA INTO SESSION STATE

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






