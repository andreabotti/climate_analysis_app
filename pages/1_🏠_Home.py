# IMPORT LIBRARIES
import os
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from ladybug.epw import EPW

from libs.fn__data import *
from libs.fn__charts import *
from libs.fn__ui import f204__custom_divider as custom_divider, absrd__epw_location_map

mapbox_access_token = 'pk.eyJ1IjoiYW5kcmVhYm90dGkiLCJhIjoiY2xuNDdybms2MHBvMjJqbm95aDdlZ2owcyJ9.-fs8J1enU5kC3L4mAJ5ToQ'



####################################################################################
# PAGE HEADER
from libs.fn__header import create_page_header


# Fixed size logo with margins
create_page_header(
    logo_width="2500px",
    logo_height="auto",
    logo_max_height="70px",
    logo_margin="65px 10px 0 0",
    logo_padding="0px",
)


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


    col_1, spacing, col_2 = st.columns([5,1,4])

    with col_1:

        st.markdown("##### App sections")
        st.markdown('Use the top navigation bar to move between sections. Each page builds on the EPW data you load here.')
        st.markdown("""
        **🏠 Home** </br>
        You are here. Upload or select EPW/STAT files from the archive (TMYx for Italy and Europe), or from EpwMap, Climate.OneBuilding, or your own files. All other pages use the files you add to the analysis list.
        </br>

        **📊 Summary Stats** </br>
        View dry-bulb temperature statistics: yearly, monthly, and daily min/max, plus daytime and nighttime extremes. Tables are colour-coded and exportable.

        **⏱ Hourly Charts** </br>
        Explore hourly data with line charts, heatmaps, sunpath diagrams, and wind roses. Filter by analysis period, bin values, and compare multiple locations side by side.

        **☀️ PV – PVGIS** </br>
        Estimate PV production using the European Commission’s PVGIS API. Get monthly energy, irradiance, and standard deviation bands. Supports multiple locations from your EPW set.

        **🏆 PV – LEED** </br>
        LEED EA Renewable Energy–aligned PV yield estimates via NREL PVWatts V8. Enter building energy use to compute renewable fraction and estimated LEED points.
        """, unsafe_allow_html=True)

    custom_divider(spacing_above_px=15, spacing_below_px=5)
    with col_2:
        st.markdown("##### EPW data sources")
        st.markdown('This app allows to visualise variables from weather data in EnergyPlus Weather Format (EPW). \
                    For more information visit [this link](https://designbuilder.co.uk/cahelp/Content/EnergyPlusWeatherFileFormat.htm) \
                    or [this link](https://climate.onebuilding.org/papers/EnergyPlus_Weather_File_Format.pdf)')
        st.markdown("""
        You can find EPW weather files at:
        - [Ladybug EpwMap](https://www.ladybug.tools/epwmap/) — interactive map for global locations
        - [Climate.OneBuilding](https://climate.onebuilding.org/) — TMY and other datasets worldwide
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




# Dictionary mapping short codes to full region names (HASC/statoids)
ITA_regions_dict = {
    "AB": "Abruzzo",
    "BC": "Basilicata",
    "CM": "Campania",
    "ER": "Emilia-Romagna",
    "FV": "Friuli-Venezia Giulia",
    "LB": "Calabria",
    "LG": "Liguria",
    "LM": "Lombardia",
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
    "VN": "Veneto",
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
    st.session_state['epw_input_choice'] = "Choose from App Archive"
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
                    "Choose from App Archive",
                    "Upload EPW files",
                    ),
                help='https://www.ladybug.tools/epwmap/',
                index=0,
            )

            # Blocked option: shown at bottom, non-selectable
            st.caption("Load ZIP files from AWS S3 Bucket")
            st.markdown(
                '<p style="font-size:0.8rem;color:#888;margin-top:-8px;">'
                '<em>Coming soon — not available</em></p>',
                unsafe_allow_html=True,
            )



        ### 1. Upload EPW files: tabs on right column ###
        if st.session_state['epw_input_choice'] == "Upload EPW files":
            with col_2:
                upload_tabs = st.tabs([
                    "EpwMap ZIP",
                    "Climate.OneBuilding ZIP",
                    "Direct EPW files",
                ])

                with upload_tabs[0]:
                    zip_data_list = st.file_uploader(
                        'Upload ZIP files from EpwMap', type='zip',
                        accept_multiple_files=True,
                        help="Download from https://www.ladybug.tools/epwmap/ then upload ZIP files here.",
                        key="epwmap_zip",
                    )
                    if zip_data_list:
                        for zip_data in zip_data_list:
                            if zip_data.name not in st.session_state['uploaded_files']:
                                zip_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{zip_data.name}')
                                zip_file_path.parent.mkdir(parents=True, exist_ok=True)
                                zip_file_path.write_bytes(zip_data.read())
                                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                                    for file_name in zip_ref.namelist():
                                        if file_name.endswith('.epw'):
                                            epw_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{file_name}')
                                            with zip_ref.open(file_name) as epw_file, epw_file_path.open('wb') as out_file:
                                                out_file.write(epw_file.read())
                                            epw_object = EPW(epw_file_path)
                                            st.session_state['uploaded_files'].append(file_name)
                                            st.session_state['epw_names'].append(epw_object)
                                            add_to_epw_files_list(file_name, epw_object, epw_file_path)
                    st.markdown('_Download by visiting [EpwMap](https://www.ladybug.tools/epwmap/) or use the iframe below, then upload ZIP files above._')
                    st.components.v1.iframe("https://www.ladybug.tools/epwmap/", height=600)

                with upload_tabs[1]:
                    zip_data_list = st.file_uploader(
                        'Upload ZIP files from Climate.OneBuilding', type='zip',
                        accept_multiple_files=True,
                        help="Download from https://climate.onebuilding.org/ then upload ZIP files here.",
                        key="climate_zip",
                    )
                    if zip_data_list:
                        for zip_data in zip_data_list:
                            if zip_data.name not in st.session_state['uploaded_files']:
                                zip_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{zip_data.name}')
                                zip_file_path.parent.mkdir(parents=True, exist_ok=True)
                                zip_file_path.write_bytes(zip_data.read())
                                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                                    for file_name in zip_ref.namelist():
                                        if file_name.endswith('.epw'):
                                            epw_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{file_name}')
                                            with zip_ref.open(file_name) as epw_file, epw_file_path.open('wb') as out_file:
                                                out_file.write(epw_file.read())
                                            epw_object = EPW(epw_file_path)
                                            st.session_state['uploaded_files'].append(file_name)
                                            st.session_state['epw_names'].append(epw_object)
                                            add_to_epw_files_list(file_name, epw_object, epw_file_path)
                    st.markdown('_Download by visiting [Climate.OneBuilding](https://climate.onebuilding.org/) or use the iframe below, then upload ZIP files above._')
                    st.components.v1.iframe("https://climate.onebuilding.org/default.html", height=600)

                with upload_tabs[2]:
                    epw_data_list = st.file_uploader('Upload EPW files', type='epw', accept_multiple_files=True, key="direct_epw")
                    if epw_data_list:
                        for epw_data in epw_data_list:
                            if epw_data.name not in st.session_state['uploaded_files']:
                                epw_file_path = pathlib.Path(f'{LOCAL_ROOT_PATH}/{epw_data.name}')
                                epw_file_path.parent.mkdir(parents=True, exist_ok=True)
                                epw_file_path.write_bytes(epw_data.read())
                                epw_object = EPW(epw_file_path)
                                st.session_state['uploaded_files'].append(epw_data.name)
                                st.session_state['epw_names'].append(epw_object)
                                add_to_epw_files_list(epw_data.name, epw_object, epw_file_path)





        ### 2. Handling EPW file selection from archive ###
        elif st.session_state['epw_input_choice'] == "Choose from App Archive":

            with col_2:
                col_menu_1, spacing, col_menu_2, spacing= st.columns([20, 5, 50, 30])

                with col_menu_1:
                    # Country selection
                    country_options = sorted(df_locations_ITA['Country'].dropna().unique())
                    selected_country = st.selectbox("Select a country:", country_options)
                    filtered_df = df_locations_ITA[df_locations_ITA["Country"] == selected_country]

                with col_menu_1:
                    # State selection: show full name alongside shortcode, e.g. "Emilia-Romagna (ER)"
                    state_options = sorted(filtered_df['State'].dropna().unique())
                    selected_state = st.selectbox(
                        "Select a State/Region:",
                        state_options,
                        format_func=lambda s: f"{ITA_regions_dict.get(s, s)} ({s})",
                    )
                    filtered_df_state = filtered_df[filtered_df['State'] == selected_state]

                selected_state_fullname = ITA_regions_dict.get(selected_state, selected_state)

                with col_menu_1:
                    # Weather station selection
                    station_options = sorted(filtered_df_state['City/Station'].dropna().unique())
                    selected_station = st.selectbox("Select a Weather Station:", station_options)

                # Pull WMO from the CSV (best match key)
                station_rows = filtered_df_state[filtered_df_state['City/Station'] == selected_station]
                wmo_val = None
                if not station_rows.empty:
                    wmo_series = station_rows["WMO"].dropna().astype(str).str.strip()
                    wmo_val = wmo_series.iloc[0] if len(wmo_series) else None

                # Scan local archive folder for all datasets for that station
                local_entries = list_local_archive_epws(
                    local_root=LOCAL_ROOT_PATH,   # "./data/"
                    country=selected_country,
                    state=selected_state,
                    city=selected_station,
                    wmo=wmo_val,
                )

                with col_menu_2:
                    # custom_divider(spacing_above_px=15, spacing_below_px=5)
                    st.markdown(''); st.markdown(''); st.markdown(''); st.markdown(''); st.markdown('')
                    st.markdown(f"###### State/Region: **{selected_state_fullname}**")
                    st.markdown(f"###### Station: **{selected_station}**")
                    # st.markdown(f"Local archive folder: {os.path.join(LOCAL_ROOT_PATH, selected_country, selected_state)}")

                    if not local_entries:
                        st.warning("No matching EPW files found in the local archive folder for this station.")
                    else:
                        # Build nice labels for selection
                        option_labels = []
                        label_to_entry = {}
                        for e in local_entries:
                            has_stat = " +STAT" if e.stat_path else ""
                            label = f"{e.period_label} — {e.file_name}{has_stat}"
                            option_labels.append(label)
                            label_to_entry[label] = e

                        selected_label = st.radio(
                            "Select a dataset to add:",
                            options=option_labels,
                            index=0 if option_labels else None,
                            key="archive_dataset_radio",
                        )

                        if st.button("Add selected dataset to the list") and selected_label:
                            entry = label_to_entry[selected_label]
                            epw_file_path = pathlib.Path(entry.epw_path)

                            already = any(x["epw_file_path"] == epw_file_path for x in st.session_state["epw_files_list"])
                            if not already:
                                epw_obj = get_epw_from_url_or_path(epw_file_path, url=None)  # local only
                                if epw_obj:
                                    st.session_state['archive_files'].append(entry.file_name)
                                    st.session_state['epw_names'].append(epw_obj)
                                    add_to_epw_files_list(entry.file_name, epw_obj, epw_file_path)
                            st.rerun()
















    with col_1:

        # custom_divider(spacing_above_px=15, spacing_below_px=5)
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
                
            custom_divider(spacing_above_px=15, spacing_below_px=5)
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
        custom_divider(spacing_above_px=15, spacing_below_px=5)
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
