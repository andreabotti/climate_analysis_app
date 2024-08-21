# IMPORT LIBRARIES
from fn__libraries import *
#
#
#
#
#
def create_page_header():

    ####################################################################################
    # PAGE CONFIGURATION
    st.set_page_config(
        page_title="Climate Analysis App",
        page_icon='https://app.pollination.cloud/favicon.ico',
        layout="wide"
    )


    ####################################################################################
    st.markdown(
        """<style>.block-container {padding-top: 1.5rem; padding-bottom: 1rem; padding-left: 2.5rem; padding-right: 2.5rem;}</style>""",
        unsafe_allow_html=True
    )

    # TOP CONTAINER
    top_col1, top_col2 = st.columns([4, 6])
    with top_col1:
        st.markdown("## Climate Analysis App")
        st.caption('Developed by AB.S.RD - https://absrd.xyz/')
    with top_col2:
        st.write('\n.')
        st.info(
            body='This app is developed using existing `ladybug-charts` library modules and charts  \t'
                '| [source code](https://github.com/pollination-apps/weather-report) | [support forum](https://discourse.pollination.cloud/c/apps/11) |'
        )

    st.markdown('---')