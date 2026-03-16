import streamlit as st
import numpy as np
import warnings

# Suppress FutureWarnings from ladybug_charts ('H' deprecated → use 'h')
warnings.filterwarnings("ignore", category=FutureWarning, module="ladybug_charts")

# NumPy 2.0 compatibility: ladybug_pandas still uses deprecated np.float_
if not hasattr(np, "float_"):
    np.float_ = np.float64

# st.set_page_config(
#     page_title            = "Climate Analysis App",
#     page_icon             = "🌍",
#     layout                = "wide",
#     initial_sidebar_state = "expanded",
# )

# Initialize session state so all pages can access it (Home populates these)
if "epw_files_list" not in st.session_state:
    st.session_state["epw_files_list"] = []
if "epw_names" not in st.session_state:
    st.session_state["epw_names"] = []
if "epw_files" not in st.session_state:
    st.session_state["epw_files"] = []
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []
if "archive_files" not in st.session_state:
    st.session_state["archive_files"] = []
if "epw_input_choice" not in st.session_state:
    st.session_state["epw_input_choice"] = "Upload EPW files"

pages = [
    st.Page("pages/1_🏠_Home.py",                   title="Home",          icon="🏠"),
    st.Page("pages/2_📊_View_Summary_Statistics.py", title="Summary Stats", icon="📊"),
    st.Page("pages/3_⏱_Hourly_Charts.py",           title="Hourly Charts", icon="⏱"),
    st.Page("pages/4_☀️_PV_Gen_Estimate.py",         title="PV – PVGIS",    icon="☀️"),
    st.Page("pages/5_🏆_PV_LEED_Estimate.py",        title="PV – LEED",     icon="🏆"),
]

pg = st.navigation(pages, position="top")
pg.run()