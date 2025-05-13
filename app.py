# IMPORT LIBRARIES
import os
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from ladybug.epw import EPW

from fn__libraries import *
from fn__chart_libraries import *

mapbox_access_token = 'pk.eyJ1IjoiYW5kcmVhYm90dGkiLCJhIjoiY2xuNDdybms2MHBvMjJqbm95aDdlZ2owcyJ9.-fs8J1enU5kC3L4mAJ5ToQ'



####################################################################################
# PAGE HEADER
from fn__page_header import create_page_header
create_page_header()





tabs = st.tabs([
    "i. **WELCOME**",
    "ii. UPLOAD DATA",
    "iii. REVIEW Uploaded Data",
    "iv. Extra: COLORS",
    ])





####################################################################################
# INTRODUCTION
with tabs[0]:

    st.markdown('#### Welcome to the Climate Analysis App.')

    st.markdown('This app allows to visualise variables from weather data in EnergyPlus Weather Format (EPW). \
                For more information visit [this link](https://designbuilder.co.uk/cahelp/Content/EnergyPlusWeatherFileFormat.htm) \
                or [this link](https://climate.onebuilding.org/papers/EnergyPlus_Weather_File_Format.pdf)')


    # custom_hr()

    # Sidebar or Main Page Navigation
    # st.markdown("##### Navigation")
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


















# Default EPW Paths
REMOTE_FTP_ROOT_PATH = r'https://absrd.xyz/streamlit_apps/epw_data/'
LOCAL_ROOT_PATH = './data/'

# Load location data from session state
df_locations_ITA = st.session_state['df_locations_ITA']
ITA_regions_dict = st.session_state['ITA_regions_dict']
df_locations_ITA = df_locations_ITA[['Country', 'State', 'City/Station', 'WMO', 'URL']]


# Initialize session state variables if not already present
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []  # Track uploaded files separately
if 'archive_files' not in st.session_state:
    st.session_state['archive_files'] = []  # Track archive-selected files separately
if 'epw_files_list' not in st.session_state:
    st.session_state['epw_files_list'] = []  # To store files for analysis
if 'epw_input_choice' not in st.session_state:
    st.session_state['epw_input_choice'] = "Upload EPW files"
if 'epw_names' not in st.session_state:  # Initialize epw_names to prevent KeyError
    st.session_state['epw_names'] = []  # Store EPW objects corresponding to the selected/uploaded files







with tabs[1]:

    col_1, col_2 = st.columns([3,7])

    with col_1:
        with st.container(height=160, border=False):

            # st.caption("""
            # Visit: [epwmap](https://www.ladybug.tools/epwmap/) or [climate.onebuilding](https://climate.onebuilding.org/) to download EPW weather data             
            # """)

            # Create radio button for EPW file input selection and store the choice in session state
            st.session_state['epw_input_choice'] = st.radio(
                "Choose EPW input method:",
                (
                    "Upload ZIP files from EpwMap",
                    "Upload ZIP files from Climate.OneBuilding.Org",
                    "Upload EPW files",
                    "Choose from App Archive",
                    ),
                help='https://www.ladybug.tools/epwmap/',
                index=1,
            )



        ### 1. Handling EPW file uploads ###
        if st.session_state['epw_input_choice'] == "Upload EPW files":
            epw_data_list = st.file_uploader('Upload EPW files', type='epw', accept_multiple_files=True)

            # Process uploaded files and add to analysis list
            if epw_data_list:
                for epw_data in epw_data_list:
                    if epw_data.name not in st.session_state['uploaded_files']:
                        # Save file locally in ./data/
                        epw_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{epw_data.name}')
                        epw_file_path.parent.mkdir(parents=True, exist_ok=True)
                        epw_file_path.write_bytes(epw_data.read())

                        # Create EPW object and add to session state
                        epw_object = EPW(epw_file_path)
                        st.session_state['uploaded_files'].append(epw_data.name)
                        st.session_state['epw_names'].append(epw_object)  # Add EPW object to epw_names
                        
                        # Add to analysis list
                        add_to_epw_files_list(epw_data.name, epw_object, epw_file_path)


        ### 1. Handling ZIP file uploads and extracting EPW files ###
        elif st.session_state['epw_input_choice'] == "Upload ZIP files from EpwMap":

            LOCAL_ROOT_PATH = "./data"  # Adjust this path as needed

            zip_data_list = st.file_uploader(
                'Upload ZIP files', type='zip',
                accept_multiple_files=True,
                help="Upload ZIP files containing EPW files to add to the analysis list.",
                )

            # Process uploaded ZIP files and extract EPW files
            if zip_data_list:
                for zip_data in zip_data_list:
                    if zip_data.name not in st.session_state['uploaded_files']:
                        # Define the extraction path
                        zip_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{zip_data.name}')
                        zip_file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Save the ZIP file locally
                        zip_file_path.write_bytes(zip_data.read())

                        # Extract EPW files
                        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                            for file_name in zip_ref.namelist():
                                if file_name.endswith('.epw'):
                                    epw_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{file_name}')
                                    with zip_ref.open(file_name) as epw_file, epw_file_path.open('wb') as out_file:
                                        out_file.write(epw_file.read())

                                    # Create EPW object and add to session state
                                    epw_object = EPW(epw_file_path)
                                    st.session_state['uploaded_files'].append(file_name)
                                    st.session_state['epw_names'].append(epw_object)  # Add EPW object to epw_names

                                    # Add to analysis list
                                    add_to_epw_files_list(file_name, epw_object, epw_file_path)


            with col_2:
                st.markdown(f'_Download by visiting: https://www.ladybug.tools/epwmap/ or using the nested iframe below, the upload ZIP files using the widget on this page_')
                # Embed EPWMap in an iframe
                st.components.v1.iframe(
                    "https://www.ladybug.tools/epwmap/",
                    # width=800,
                    height=600,
                    )



        ###
        elif st.session_state['epw_input_choice'] == "Upload ZIP files from Climate.OneBuilding.Org":

            with col_2:
            
                with st.container(height=600, border=False):
                    st.markdown(f'_Download by visiting: https://climate.onebuilding.org/ or using the nested iframe below, the upload ZIP files using the widget on this page_')

                    # Embed webpage climate.onebuilding.org in an iframe
                    st.components.v1.iframe(
                        "https://climate.onebuilding.org/default.html",
                        # width=800,
                        height=3000,
                        )



        ### 2. Handling EPW file selection from archive ###
        elif st.session_state['epw_input_choice'] == "Choose from App Archive":

            with col_2:
                col_menu_1, col_menu_2, col_menu_3 = st.columns([3,8,1])
                with col_menu_1:
                    # Country selection
                    country_options = df_locations_ITA['Country'].unique() 
                    selected_country = st.selectbox("Select a country:", country_options)
                    filtered_df = df_locations_ITA[df_locations_ITA["Country"] == selected_country]

                with col_menu_1:
                    # State selection
                    state_options = sorted(filtered_df['State'].unique())
                    selected_state = st.selectbox("Select a State/Region:", state_options)
                    filtered_df = filtered_df[filtered_df['State'] == selected_state]

                selected_state_fullname = ITA_regions_dict[selected_state]

                with col_menu_1:
                    # Weather station selection
                    station_options = sorted(filtered_df['City/Station'].unique())
                    selected_station = st.selectbox("Select a Weather Station:", station_options)
                    filtered_df = filtered_df[filtered_df['City/Station'] == selected_station]

                # custom_hr()


            if selected_station:
                # Get EPW file URL and local path
                selected_epw_url = filtered_df['URL'].values[0]
                selected_epw_file = selected_epw_url.replace('.zip', '.epw').rsplit('/', 1)[1]
                epw_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{selected_country}/{selected_state}/{selected_epw_file}')
                # epw_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{selected_country}/{selected_state}/{selected_epw_file}')


                with col_menu_1:
                    custom_hr()
                    st.markdown(f'State/Region: **{selected_state_fullname}**')
                    st.markdown(f'Station: **{selected_epw_file}**')

                    if st.button(f"Add file to the list"):
                        epw_obj = get_epw_from_url_or_path(epw_file_path, url=selected_epw_url)
                        if epw_obj:
                            # Add selected file to archive list and analysis list
                            st.session_state['archive_files'].append(selected_epw_file)
                            st.session_state['epw_names'].append(epw_obj)  # Add EPW object to epw_names
                            add_to_epw_files_list(selected_epw_file, epw_obj, epw_file_path)


    with col_1:

        # custom_hr()
        st.divider()
        with st.container(height=350, border=False):
            ### 3. Display the analysis list ###
            st.markdown("###### List of EPW files to be analysed")
            if st.session_state['epw_files_list']:
                for entry in st.session_state['epw_files_list']:
                    col1, col2 = st.columns([9,3])
                    with col1:
                        # st.markdown(f"**{entry['epw_file_name']}** - {entry['epw_object'].location.city}, {entry['epw_object'].location.country}")
                        st.markdown(f"{entry['epw_file_name']}")

                    with col2:
                        # Remove button for each entry in the analysis list
                        if st.button(f"Remove", key=f"remove_{entry['epw_file_name']}"):
                            st.session_state['epw_files_list'].remove(entry)
                            st.rerun()

















with tabs[2]:

    col_info, col_maps = st.columns([2,6])

    with col_info:


        # Display uploaded or selected EPW files if available
        # all_files = st.session_state['uploaded_files'] + st.session_state['archive_files']
        epw_names = [item['epw_file_name'] for item in st.session_state['epw_files_list'] ]
        epw_files = [item['epw_object'] for item in st.session_state['epw_files_list'] ]
        epw_file_paths = [item['epw_file_path'] for item in st.session_state['epw_files_list'] ]

        stat_names = [item['epw_file_name'] for item in st.session_state['epw_files_list'] ]



    if epw_names:
        with col_info:
            st.markdown('')
            st.markdown(f'_{len(epw_names)} EPW files have been processed_')
            for f in epw_names:
                st.write(f)
                
            custom_hr()
            # with st.expander('EPW dict'):
            #     st.write( st.session_state['epw_files_list'] )


        with col_maps:
            if epw_names:  # Check if epw_names is not empty
                map_cols = st.columns(len(epw_names))
                for col, epw_name, epw_file in zip(map_cols, epw_files, epw_names):
                    with col:
                        st.markdown(f'###### {epw_name.location.city}   |   Country: {epw_name.location.country}')
                        # st.caption(epw_file)
                        location = pd.DataFrame([np.array([epw_name.location.latitude, epw_name.location.longitude], dtype=np.float64)], columns=['latitude', 'longitude'])

                        # Uncomment below line if `absrd__epw_location_map` is a valid function for generating maps
                        map_plot = absrd__epw_location_map(
                            data=location, col_lat='latitude', col_lon='longitude',
                            zoom_level=5, chart_height=250, chart_width=100, dot_size=12000, dot_color=[255, 0, 0], dot_opacity=100,
                            )
                        map_plot = absrd__epw_location_map(
                            data=location, col_lat='latitude', col_lon='longitude',
                            zoom_level=8, chart_height=250, chart_width=100, dot_size=1500, dot_color=[255, 0, 0], dot_opacity=100,
                            )
    else:
        st.markdown('##### No EPW files have been processed or selected.')


    # st.write( st.session_state['epw_files_list'] )



# st.session_state['epw_files_list'] = epw_files_list
st.session_state['epw_names'] = epw_names
st.session_state['epw_files'] = epw_files






with tabs[3]:

    def convert_to_css_rgba(color_tuple):
        r, g, b, a = color_tuple
        return f"rgba({r}, {g}, {b}, {a / 255:.2f})"  # Normalize alpha

    st.header("Color Set Preview")


    for name, color_list in colorsets.items():
        custom_hr()
        col1, col2 = st.columns([1,5])
        col1.markdown(f"{name}")
        # st.write("Raw color values:", color_list)

        boxes_html = ""
        for color_tuple in color_list:
            rgba_str = convert_to_css_rgba(color_tuple)
            boxes_html += f"""
                <div style="
                    width: 24px;
                    height: 24px;
                    background: {rgba_str};
                    display: inline-block;
                    margin-right: 0px;
                    border: 1px solid #ccc;
                    border-radius: 0px;
                " title="{rgba_str}"></div>
            """
        col2.markdown(boxes_html, unsafe_allow_html=True)
