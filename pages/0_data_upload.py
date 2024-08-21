# IMPORT LIBRARIES
import os
import pathlib
import streamlit as st
from ladybug.epw import EPW
from fn__libraries import *
# from template_graphs import *

mapbox_access_token = 'pk.eyJ1IjoiYW5kcmVhYm90dGkiLCJhIjoiY2xuNDdybms2MHBvMjJqbm95aDdlZ2owcyJ9.-fs8J1enU5kC3L4mAJ5ToQ'

####################################################################################
# PAGE HEADER
from fn__page_header import create_page_header
create_page_header()

####################################################################################
# Default EPW Path
DEFAULT_EPW_PATH = './assets/sample.epw'

# Initialize session state if not already present
if 'epw_files' not in st.session_state:
    st.session_state['epw_files'] = []
    st.session_state['epw_names'] = []

# EPW file uploader
epw_data_list = st.file_uploader('Upload EPW files', type='epw', accept_multiple_files=True)

# Clear previously uploaded EPW files after upload
if st.button('Clear / Update Uploaded EPW Files'):
    st.session_state['epw_files'].clear()
    st.session_state['epw_names'].clear()
    st.experimental_rerun()  # Force rerun the app to reflect changes immediately


custom_hr()

# Process uploaded files
if epw_data_list:
    uploaded_files = [epw_data.name for epw_data in epw_data_list]

    # Avoid duplicates by checking if the file is already in the session state
    for epw_data in epw_data_list:
        if epw_data.name not in st.session_state['epw_files']:
            epw_file = pathlib.Path(f'./data/{epw_data.name}')
            st.session_state['epw_files'].append(epw_data.name)
            epw_file.parent.mkdir(parents=True, exist_ok=True)
            epw_file.write_bytes(epw_data.read())
            st.session_state['epw_names'].append(EPW(epw_file))

# Fallback to default EPW if no files uploaded and default file exists
if not st.session_state['epw_names'] and os.path.exists(DEFAULT_EPW_PATH):
    st.session_state['epw_names'].append(EPW(DEFAULT_EPW_PATH))
    st.session_state['epw_files'].append('sample.epw')

# Check if there are any EPW files in the session state
if not st.session_state['epw_names']:
    st.error("Please upload one or more EPW files.")
    st.stop()

# Retrieve the EPW files and names from the session state
epw_files = st.session_state['epw_files']
epw_names = st.session_state['epw_names']



# Display uploaded EPW files if available
if epw_files:
    st.markdown(f'###### {len(epw_files)} EPW files have been uploaded')
    for f in epw_files:
        st.caption(f)
else:
    st.markdown('##### No EPW files have been uploaded')


custom_hr()

# Display maps
with st.expander(label='Map View of uploaded EPW files'):
    # st.markdown('#### Map View of uploaded EPW files')
    map_cols = st.columns(len(epw_names))

    for col, epw_name, epw_file in zip(map_cols, epw_names, epw_files):
        with col:
            st.markdown(f'###### {epw_name.location.city}   |   Country: {epw_name.location.country}')
            st.caption(epw_file)
            location = pd.DataFrame([np.array([epw_name.location.latitude, epw_name.location.longitude], dtype=np.float64)], columns=['latitude', 'longitude'])
            st.map(location, zoom=9, use_container_width=True)
