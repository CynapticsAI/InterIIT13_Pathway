import streamlit as st
from Helper.HybridFL_Setup import *

st.set_page_config(
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",

)

# Set up the directory for uploaded files
UPLOAD_DIR = "../DocumentStore/files-for-indexing"
absolute_upload_dir = os.path.abspath(UPLOAD_DIR)
# os.makedirs(absolute_upload_dir, exist_ok=True)  

#st.set_page_config(layout="wide")

# Function to simulate text streaming
def string_streamer(text, chunk_size=1, delay=0.005):
    for i in range(0, len(text), chunk_size):
        yield text[:i + chunk_size]
        time.sleep(delay)

# Function to process user query
def generate_response(query):
    return query

# Set up the title and chat interface
st.header("💬 HybridFL ChatBot", divider=True)

# Initialize session state for messages and context
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "context" not in st.session_state:
    st.session_state["context"] = "Random context: " + random.choice(["The sky is blue.", "Cats are great.", "Technology is evolving.", "The weather is nice today."])

# Display existing messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# User input box
if user_input := st.chat_input("Type your message here..."):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Generate a response using the function
    st.markdown("""
        <style>
            .stSpinner {
                margin-left: 20px; /* Adjust this value to your preference */
            }
        </style>
    """, unsafe_allow_html=True)
    with st.spinner("Generating response..."):
        response = qna(user_input)

    # Simulate streaming response
    with st.chat_message("assistant"):
        response_container = st.empty()
        for streamed_text in string_streamer(response):
            response_container.write(streamed_text)

    # Append assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a dropdown for model selection and an input field for API key in the sidebar
# Add a dropdown for model selection and an input field for API key in the sidebar
with st.sidebar:
    # File upload section (unchanged from your code)
    # st.header("File Upload")

    # Initialize session state for file management
    # Initialize session state for file management
    if "filelist" not in st.session_state:
        st.session_state.filelist = []
        for root, dirs, files in os.walk(UPLOAD_DIR):
            for file in files:
                filename = os.path.join(root, file)
                if filename not in st.session_state.filelist:
                    st.session_state.filelist.append(filename)

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1

    # File uploader
    st.header("File Upload")
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["txt", "pdf"], 
        accept_multiple_files=True, 
        key=st.session_state["uploader_key"], 
        label_visibility="collapsed"
    )

    # Save uploaded files and update the filelist
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(absolute_upload_dir, uploaded_file.name)
            if save_path not in st.session_state.filelist:  # Avoid duplicates
                st.session_state.filelist.append(save_path)
            try:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            except Exception as e:
                st.error(f"Error saving file: {e}")
        # Clear the uploader after processing
        st.session_state["uploader_key"] += 1
        st.rerun()

    # Function to show files with checkboxes
    def show_files():
        for file_path in st.session_state.filelist:
            file_name = os.path.basename(file_path)
            st.session_state['source_files'][file_path] = st.checkbox(file_name, key=f"checkbox_{file_path}")

    # Function to remove selected files
    def remove_files():
        files_to_remove = [file for file, selected in st.session_state['source_files'].items() if selected]
        for file_path in files_to_remove:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    st.session_state.filelist.remove(file_path)
                    st.session_state['source_files'].pop(file_path)
            except Exception as e:
                st.error(f"Error deleting file: {e}")
        st.rerun()

    # Display the filelist and allow deletion
    if 'source_files' not in st.session_state:
        st.session_state['source_files'] = {}

    with st.expander("See Files"):
        show_files()
        if st.button("Delete Selected Files"):
            remove_files()
    # print(st.session_state.filelist)

    # Dropdown to select OpenAI model
    model_choice = st.selectbox(
        "Select OpenAI Model",
        options=["gpt-3.5-turbo-0125","gpt-4-turbo", "o1-preview", "gpt-4o-mini","gpt-4o"],
        index=3  # Default to GPT-4o-mini
    )
    st.session_state.model_choice = model_choice

    # Text input for entering the API key
    # api_key = st.text_input(
    #     "Enter OpenAI API Key",
    #     type="password",  # Mask the input
    #     help="Please enter your OpenAI API key here."
    # )
    # st.session_state.api_key = api_key
    api_key = st.text_input(
        "Enter OpenAI API Key",
        type="password",  # Mask the input
        help="Please enter your OpenAI API key here."
    )

    # Add a button to update the API key
    if st.button("Update API Key"):
        st.session_state.api_key = api_key
        st.success("API Key updated successfully.")

    # # Button to open modal (already present in your code)
    # # Custom CSS to style the button
    # st.markdown("""
    #     <style>
    #         /* Target primary button */
    #     button[kind="primary"] {
    #         width: 290px; /* Set the width of the button */
    #         background-color: #ff6c6c; /* Change the background color */
    #         color: white; /* Text color */
    #         font-size: 20px; /* Font size */
    #         border-radius: 5px; /* Rounded corners */
    #         height: 50px; /* Button height */
    #         border: none; /* Remove border */
    #         cursor: pointer; /* Pointer cursor on hover */
    #     }

    #     button[kind="primary"]:hover {
    #         background-color: white;
    #         color: #ff6c6c
    #     }
    #     </style>
    # """, unsafe_allow_html=True)
    # open_modal = st.button("Reasoning",type = 'primary')
    # if open_modal:
    #     modal.open()

    # if modal.is_open():
    #     with modal.container():
    #         # Assuming you are using session state to track the responses and queries
    #         subquery_list = st.session_state["subquery_list"]
    #         current_response_list = st.session_state["current_response_list"]
    #         tools_dict = st.session_state["tools_dict"]
    #         if tools_dict:
    #             st.title("Tools Used")
    #             for data in tools_dict:
    #                 st.subheader(f"Tool Name:")
    #                 st.write(data['tool_name'])
    #                 st.subheader(f"Arguments:")
    #                 st.write(data['arguments'])
    #         if current_response_list:
    #             st.title("Retrievals")
    #             tablist = [str(i+1) for i in range(len(current_response_list))]

    #             # Create tabs using numeric labels
    #             tabs = st.tabs(tablist)

    #             for i, response in enumerate(current_response_list):
    #                 with tabs[i]:
    #                     st.subheader("Current Response")
    #                     if response.lower() == '0':
    #                         st.write("Initially there is no response")
    #                     else:
    #                         st.text_area(
    #                             label='', 
    #                             value=response, 
    #                             label_visibility='hidden', 
    #                             height=200, 
    #                             disabled=True,
    #                             key=f"current_response_{i}"
    #                         )

    #                     # Display subqueries
    #                     st.subheader("Subqueries")
    #                     inner_dict = subquery_list[i]

    #                     for subquery_index, (subquery, contexts) in enumerate(inner_dict.items()):
    #                         # Input for each subquery
    #                         st.text_input(
    #                             label=f"Subquery {subquery_index + 1}", 
    #                             value=subquery, 
    #                             disabled=False, 
    #                             key=f"subquery_input_{i}_{subquery_index}"
    #                         )

    #                         # Button for regenerating response
    #                         if st.button(f"Regenerate Response {subquery_index + 1}", key=f"regen_button_{i}_{subquery_index}"):
    #                             updated_response = regenerated_answer(prev_response=subquery, regenerated_query=subquery)
    #                             st.success(f"Updated Response: {updated_response}")

    #                         # Expander for contexts
    #                         with st.expander(f"Context Retrieved for Subquery {subquery_index + 1}"):
    #                             for context_index, context in enumerate(contexts):
    #                                 st.text_area(
    #                                     label=f"Context {context_index + 1}", 
    #                                     value=context.page_content, 
    #                                     label_visibility='hidden', 
    #                                     height=500, 
    #                                     disabled=True,
    #                                     key=f"context_{i}{subquery_index}{context_index}")