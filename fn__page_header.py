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
    col_logo, col_title, col_info = st.columns([1,4,5])

    with col_logo:
        st.image('./img/logo_small.png', use_column_width=True)

    with col_title:
        st.markdown("# Climate Analysis App ")
        st.markdown("##### Exploration of weather data in EPW format")
        # st.caption('Developed by AB.S.RD - https://absrd.xyz/')

    with col_info:
        st.write('\n')
        # st.write('')
        st.caption('Developed by AB.S.RD - https://absrd.xyz/')
        st.info(
            icon="ℹ️",
            body='This app is developed using existing `ladybug-charts` library modules and charts \t'
                '[(see source code)](https://github.com/pollination-apps/weather-report)'
                #  '' [support forum](https://discourse.pollination.cloud/c/apps/11) |'
        )

        # st.markdown('**Info**')
        # st.markdown(
        #     body='**Info** [ This app is developed using existing `ladybug-charts` library modules and charts ]  \t'
        #         '| [source code](https://github.com/pollination-apps/weather-report) | [support forum](https://discourse.pollination.cloud/c/apps/11) |'
        # )

    custom_hr()