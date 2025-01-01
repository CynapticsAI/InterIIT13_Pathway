import streamlit as st

st.set_page_config(
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",

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

# Custom CSS for card-like appearance
st.markdown(
    """
<style>
.card {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    transition: transform 0.3s ease;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.card:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 8px rgba(0,0,0,0.15);
}
.card img {
    max-width: 100%;
    border-radius: 10px;
    margin-bottom: 15px;
}
h1 {
            margin-top: -40px; /* Adjust the value to move the title up */
        }
</style>
""",
    unsafe_allow_html=True,
)

# Page title
# st.title("Agentic RAG")
st.markdown("<h1 style='text-align: center;'>Agentic RAG</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; margin:-20px ;color:grey;'>Choose your pipeline</h1>", unsafe_allow_html=True)
# st.subheader("Choose your pipeline")

# Create three columns for cards
col1,col2, col3 = st.columns(3)

# Science Card
with col1:
    # st.image("image_1.jpg")
    st.header(":red[FLARE]")
    # st.subheader(":grey[(Closed Source)]")
    st.markdown("<h5 style='color:grey;margin-top:-20px'>Closed Source - General Purpose</h5>", unsafe_allow_html=True)
    st.write(
        "A general purpose RAG pipeline built on top of pathway for efficient "
        "data processing integrated with OpenAI Swarm for Autonomous Agentic Control and decision making."
    )
    if st.button("Open Chatbot", key="science_btn"):
        st.session_state.page = "flare"
        st.switch_page(FLARE)

# Technology Card
with col2:
    # st.image("image_2.jpg")
    st.header(":red[DRAGIN]")
    # st.subheader(":grey[(Open Source)]")
    st.markdown("<h5 style='color:grey;margin-top:-20px'>Open Source - General Purpose</h5>", unsafe_allow_html=True)
    st.write(
        "A general purpose RAG pipeline built using pathway for efficient data "
        "processing integrated with a free open source LLM to provide a free-to-use efficient pipeline for RAG."
    )
    if st.button("Open Chatbot", key="tech_btn"):
        st.session_state.page = "dragin"
        st.switch_page(DRAGIN)

# Environment Card
with col3:
    # st.image("image_3.jpg")
    st.header(":red[HybridFL]")
    st.markdown("<h5 style='color:grey;margin-top:-20px'>Closed Source - Finance / Legal Purpose</h5>", unsafe_allow_html=True)
    st.write(
        "A finance and legal specific pipeline built on top of a pathway-based hybrid rag for accurate retrieval "
        "integrated with OpenAI swarm for autonomous agentic control and decision making."
    )
    if st.button("Open Chatbot", key="env_btn"):
        st.session_state.page = "fishyrag"
        st.switch_page(HYBRIDFL)


