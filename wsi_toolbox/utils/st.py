from contextlib import contextmanager

import streamlit as st

HORIZONTAL_STYLE = """
<style class="hide-element">
    /* Hides the style container and removes the extra spacing */
    .element-container:has(.hide-element) {
        display: none;
    }
    /*
        The selector for >.element-container is necessary to avoid selecting the whole
        body of the streamlit app, which is also a stVerticalBlock.
    */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) {
        display: flex;
        flex-direction: row !important;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: baseline;
    }
    /* Buttons and their parent container all have a width of 704px, which we need to override */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) div {
        width: max-content !important;
    }
    /* Selectbox container */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) div[data-testid="stSelectbox"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    /* Selectbox label */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) div[data-testid="stWidgetLabel"] {
        margin-bottom: 0 !important;
        padding-right: 0.5rem !important;
    }
    /* Selectbox input container */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) div[data-baseweb="select"] {
        min-width: 120px !important;
    }
    /* Selectbox dropdown */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) div[role="listbox"] {
        min-width: 120px !important;
    }
</style>
"""

@contextmanager
def st_horizontal():
    st.markdown(HORIZONTAL_STYLE, unsafe_allow_html=True)
    with st.container():
        st.markdown('<span class="hide-element horizontal-marker"></span>', unsafe_allow_html=True)
        yield
