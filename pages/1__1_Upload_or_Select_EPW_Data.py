# IMPORT LIBRARIES
import os
import pathlib
import requests
import streamlit as st
from ladybug.epw import EPW
import pandas as pd
import numpy as np
from fn__libraries import *  # Assuming fn__libraries contains your custom functions

# PAGE HEADER
from fn__page_header import create_page_header
create_page_header()

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


# Create radio button for EPW file input selection and store the choice in session state
st.session_state['epw_input_choice'] = st.sidebar.radio(
    "Choose EPW input method:", 
    ("Upload EPW files", "Choose from Archive"), 
    index=1,
)


# Function to download file from a URL and save locally
def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        st.error(f"Failed to download the EPW file from {url}")
        return False

# Function to create an EPW object from a URL or local file path
def get_epw_from_url_or_path(epw_file_path, url=None):
    # If the path is a URL, download it locally first
    if url:
        local_filename = epw_file_path.name  # Get filename from path
        if not epw_file_path.exists():  # Download if not already present
            success = download_file(url, epw_file_path)
            if not success:
                st.error(f"Unable to download EPW file from {url}")
                return None
    # Create EPW object using local path
    return EPW(epw_file_path)



# Function to add a file to the analysis list
def add_to_epw_files_list(epw_file_name, epw_object, epw_file_path):
    if epw_file_name not in [item['epw_file_name'] for item in st.session_state['epw_files_list']]:
        st.session_state['epw_files_list'].append({
            'epw_file_name': epw_file_name,
            'epw_object': epw_object,
            'epw_file_path': epw_file_path,
            'stat_file_name': epw_file_name.replace('.epw', '.stat'),
            'stat_file_path': str(epw_file_path).replace('.epw', '.stat'),
            })
        st.success(f"Added {epw_file_name} to Analysis List.")
    else:
        st.warning(f"{epw_file_name} is already in the analysis list.")


### 1. Handling EPW file uploads ###
if st.session_state['epw_input_choice'] == "Upload EPW files":
    st.sidebar.write("Upload EPW files to add to the analysis list.")
    epw_data_list = st.sidebar.file_uploader('Upload EPW files', type='epw', accept_multiple_files=True)

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


### 2. Handling EPW file selection from archive ###
elif st.session_state['epw_input_choice'] == "Choose from Archive":
    with st.sidebar:
        st.markdown("---")

        col_menu_1, col_menu_2 = st.columns([2,3])
        with col_menu_1:
            # Country selection
            country_options = df_locations_ITA['Country'].unique() 
            selected_country = st.selectbox("Select a country:", country_options)
            filtered_df = df_locations_ITA[df_locations_ITA["Country"] == selected_country]

        with col_menu_2:
            # State selection
            state_options = sorted(filtered_df['State'].unique())
            selected_state = st.selectbox("Select a State/Region:", state_options)
            filtered_df = filtered_df[filtered_df['State'] == selected_state]

        selected_state_fullname = ITA_regions_dict[selected_state]

        # Weather station selection
        station_options = sorted(filtered_df['City/Station'].unique())
        selected_station = st.selectbox("Select a Weather Station:", station_options)
        filtered_df = filtered_df[filtered_df['City/Station'] == selected_station]

        custom_hr()
        st.caption(f'State/Region: {selected_state_fullname}')

        if selected_station:
            # Get EPW file URL and local path
            selected_epw_url = filtered_df['URL'].values[0]
            selected_epw_file = selected_epw_url.replace('.zip', '.epw').rsplit('/', 1)[1]
            epw_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{selected_country}/{selected_state}/{selected_epw_file}')
            # epw_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{selected_country}/{selected_state}/{selected_epw_file}')


            # Display the "Add to Analysis List" button
            st.caption(f'{selected_epw_file}')
            if st.button(f"Add file to the list"):
                epw_obj = get_epw_from_url_or_path(epw_file_path, url=selected_epw_url)
                if epw_obj:
                    # Add selected file to archive list and analysis list
                    st.session_state['archive_files'].append(selected_epw_file)
                    st.session_state['epw_names'].append(epw_obj)  # Add EPW object to epw_names
                    add_to_epw_files_list(selected_epw_file, epw_obj, epw_file_path)



### 3. Display the analysis list ###
st.markdown("##### List of loaded EPW files")
if st.session_state['epw_files_list']:
    for entry in st.session_state['epw_files_list']:
        col1, col2, spacing = st.columns([4,2,2])
        with col1:
            st.markdown(f"**{entry['epw_file_name']}** - {entry['epw_object'].location.city}, {entry['epw_object'].location.country}")
        with col2:
            # Remove button for each entry in the analysis list
            if st.button(f"Remove file", key=f"remove_{entry['epw_file_name']}"):
                st.session_state['epw_files_list'].remove(entry)
                st.rerun()


custom_hr()


# Display uploaded or selected EPW files if available
# all_files = st.session_state['uploaded_files'] + st.session_state['archive_files']
epw_names = [item['epw_file_name'] for item in st.session_state['epw_files_list'] ]
epw_files = [item['epw_object'] for item in st.session_state['epw_files_list'] ]
epw_file_paths = [item['epw_file_path'] for item in st.session_state['epw_files_list'] ]

stat_names = [item['epw_file_name'] for item in st.session_state['epw_files_list'] ]




if epw_names:
    col_info, col_maps = st.columns([3,6])

    with col_info:
        st.markdown(f'###### {len(epw_names)} EPW files have been processed')
        for f in epw_names:
            st.write(f)
        
        custom_hr()
        with st.expander('EPW dict'):
            st.write( st.session_state['epw_files_list'] )

    with col_maps:
        if epw_names:  # Check if epw_names is not empty
            map_cols = st.columns(len(epw_names))
            for col, epw_name, epw_file in zip(map_cols, epw_files, epw_names):
                with col:
                    st.markdown(f'###### {epw_name.location.city}   |   Country: {epw_name.location.country}')
                    # st.caption(epw_file)
                    location = pd.DataFrame([np.array([epw_name.location.latitude, epw_name.location.longitude], dtype=np.float64)], columns=['latitude', 'longitude'])

                    # Uncomment below line if `absrd__epw_location_map` is a valid function for generating maps
                    map_plot = absrd__epw_location_map(data=location, col_lat='latitude', col_lon='longitude', zoom_level=8, chart_height=100, chart_width=100)
else:
    st.markdown('##### No EPW files have been processed or selected.')


# st.write( st.session_state['epw_files_list'] )



# st.session_state['epw_files_list'] = epw_files_list
st.session_state['epw_names'] = epw_names
st.session_state['epw_files'] = epw_files

