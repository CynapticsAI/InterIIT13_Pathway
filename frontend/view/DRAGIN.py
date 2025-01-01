import streamlit as st
import os
import random
import time
from Helper.utils import AttnWeightRAG
from streamlit_modal import Modal

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "context" not in st.session_state:
    st.session_state["context"] = "Random context: " + random.choice(
        ["The sky is blue.", "Cats are great.", "Technology is evolving.", "The weather is nice today."]
    )
if "subquery_list" not in st.session_state:
    st.session_state["subquery_list"] = []
if "current_response_list" not in st.session_state:
    st.session_state["current_response_list"] = {}
if "tools_dict" not in st.session_state:
    st.session_state["tools_dict"] = {}




config = {
    "model_name_or_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "method": "attn_prob",
    "dataset": "2wikimultihopqa",
    "data_path": "data/2wikimultihopqa",
    "generate_max_length": 1024,
    "query_formulation": "real_words",
    "retrieve_keep_top_k": 40,
    "output_dir": "../result/2wikimultihopqa_llama2_13b",
    "retriever": "BM25",
    "retrieve_topk": 3,
    "hallucination_threshold": 1.2,
    "fewshot": 6,
    "sample": 1000,
    "shuffle": False,
    "check_real_words": True,
    "es_index_name": "34051_wiki",
    "use_counter": True
}

config_path = 'config.json'
import os
if os.path.exists(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



def convert_l_to_d(contexts):
    query_list = []
    ans_list = []
    for idx, context in enumerate(contexts):
        dct = {}
        hallucinating_idx = 3 if idx == 0 else 2
        if context[hallucinating_idx]:
            dct['query'] = context[hallucinating_idx+2]
            dct['contexts'] = context[hallucinating_idx+3]
        query_list.append(dct)
        ans_list.append(context[0])
    return ans_list, query_list



# Directory setup
UPLOAD_DIR = "../DocumentStore/files-for-indexing"
absolute_upload_dir = os.path.abspath(UPLOAD_DIR)


# def load_model():
#     """Load the model."""
#     if not st.session_state["model_loaded"]:
#         st.session_state["model_loaded"] = True
#         model.retrieve = retrieve.__get__(model, AttnWeightRAG)
#         st.success("Model loaded successfully!")

def retrieve(self, query, topk=1, max_query_length=64):
    self.counter.retrieve += 1
    # complete the retriever function
    # docs = bm25_retriever.invoke("foo")
    # docs = [d.page_content for d in docs]
    # docs = client(query, k = topk)
    docs = [f"Thses are some dummy docs for {query}" for _ in range(topk)]
    return docs
# model.retrieve = retrieve.__get__(model, AttnWeightRAG)
# model.retrieve("What is Nvidia", topk=3) # for testing

def generate_dragin_output(question):
    output,context = model.inference(question, {}, f"Question: {question}\nAnswer:")
    return output,context

def string_streamer(text, chunk_size=1, delay=0.02):
    """Simulate text streaming."""
    for i in range(0, len(text), chunk_size):
        yield text[:i + chunk_size]
        time.sleep(delay)

if st.session_state["model_loaded"]:
    args = Args(**config)
    model = AttnWeightRAG(args)
    model.retrieve = retrieve.__get__(model, AttnWeightRAG)



# Sidebar
with st.sidebar:
    # File upload section
    st.header("File Upload")
    if "filelist" not in st.session_state:
        st.session_state.filelist = []
        for root, dirs, files in os.walk(UPLOAD_DIR):
            for file in files:
                filename = os.path.join(root, file)
                if filename not in st.session_state.filelist:
                    st.session_state.filelist.append(filename)

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1

    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["txt", "pdf"], 
        accept_multiple_files=True, 
        key=st.session_state["uploader_key"], 
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(absolute_upload_dir, uploaded_file.name)
            if save_path not in st.session_state.filelist:
                st.session_state.filelist.append(save_path)
            try:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            except Exception as e:
                st.error(f"Error saving file: {e}")
        st.session_state["uploader_key"] += 1
        st.rerun()

    if 'source_files' not in st.session_state:
        st.session_state['source_files'] = {}

    with st.expander("See Files"):
        for file_path in st.session_state.filelist:
            file_name = os.path.basename(file_path)
            st.session_state['source_files'][file_path] = st.checkbox(file_name, key=f"checkbox_{file_path}")

        if st.button("Delete Selected Files"):
            files_to_remove = [
                file for file, selected in st.session_state['source_files'].items() if selected
            ]
            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    st.session_state.filelist.remove(file_path)
                    st.session_state['source_files'].pop(file_path)
            st.rerun()

    # Model selection dropdown
    st.header("Model Settings")
    model_choice = st.selectbox(
        "Select OpenAI Model",
        options=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "google/gemma-2-9b-it", 
            "Qwen/Qwen2.5-14B-Instruct", 
            "Qwen/Qwen2.5-14B",
            "HuggingFaceTB/SmolLM2-360M-Instruct"
        ],
        index=4
    )

    if st.button("Select Model"):
        st.session_state.model_choice = model_choice
        if st.session_state["model_loaded"]:
            model.change_model(model_choice)
        else:
            args = Args(**config)
            model = AttnWeightRAG(args)
            model.retrieve = retrieve.__get__(model, AttnWeightRAG)
            st.session_state["model_loaded"] = True
        # with st.spinner("Loading model..."):
        #     load_model()

    # API Key Input
    api_key = st.text_input(
        "Enter Hugging Face Access Token",
        type="password",
        help="Please enter your Hugging Face Access Token here."
    )

    if st.button("Update API Key"):
        st.session_state.api_key = api_key
        from huggingface_hub import login

        # Replace 'your_token_here' with your actual Hugging Face token.
        login(token=st.session_state.api_key)

        st.success("API Key updated successfully.")

# Main Content
st.header("🚀 DRAGIN Chat Bot", divider=True)

# Chat interface
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

if user_input := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.spinner("Generating response..."):
        # if st.session_state["model_loaded"]:
        response, context = generate_dragin_output(user_input)


        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        response_container = st.empty()
        for streamed_text in string_streamer(response):
            response_container.write(streamed_text)

# Modal for reasoning
modal = Modal("💬Thoughts", key="demo-modal", padding=20, max_width=744)
if st.sidebar.button("Reasoning", type="primary"):
    modal.open()

if modal.is_open():
    with modal.container():
        st.title("Tools Used")
        tools_dict = st.session_state["tools_dict"]
        for data in tools_dict:
            st.subheader("Tool Name:")
            st.write(data['tool_name'])
            st.subheader("Arguments:")
            st.write(data['arguments'])
