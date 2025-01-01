import streamlit as st
from streamlit_navigation_bar import st_navbar

HomePage = st.Page(
    page = 'view/HomePage.py',
    title = 'Home Page',
    default = True,
)
FLARE = st.Page(
    page = 'view/FLARE.py',
    title = 'FLARE',
)
DRAGIN = st.Page(
    page = 'view/DRAGIN.py',
    title = 'DRAGIN',
)
HYBRIDFL = st.Page(
    page = 'view/HYBRIDFL.py',
    title = 'HYBRIDFL',
)

# pg = st.navigation({
#     "":[HomePage],
#     "General Purpose RAG":[FLARE,DRAGIN],
#     "Financial / Legal RAG":[HYBRIDFL],
# })
pg = st.navigation([HomePage,FLARE,DRAGIN,HYBRIDFL])

# st.set_page_config(
#     # page_icon="🧊",
#     layout="wide",
#     initial_sidebar_state="collapsed" 

# )

pg.run()