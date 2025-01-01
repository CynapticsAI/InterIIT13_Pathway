
import pathway as pw

import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

import PyPDF2

import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import networkx as nx

from sentence_transformers import SentenceTransformer


from langchain.text_splitter import CharacterTextSplitter
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import networkx as nx
import time

#Here a blank graph is created and passed, in our main code a pre built graph is passed and modified whenever a document is added
G = nx.Graph()
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def count_lines_in_file(file_path):
    """
    Counts the number of lines in a file.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        int: The number of lines in the file.
    """
    with open(file_path, 'r') as file:
        # Count the number of lines
        line_count = sum(1 for line in file)
    return line_count

def extract_data_and_time(file_path,lines):
    """
    Extracts 'data' and 'time' fields from a JSON file containing separate JSON objects.
    Also, chunks the 'data' text into smaller parts using Langchain.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        list: A list of dictionaries containing 'data' and 'time', with 'data' chunked.
    """
    with open(file_path, 'r') as file:
        for _ in range(lines - 1):
            next(file)
        content = file.read()
    
    json_objects = content.strip().split("}\n{")
    json_objects = [f"{{{obj}}}" for obj in json_objects]
    json_objects[0] = json_objects[0][1:]  
    json_objects[-1] = json_objects[-1][:-1]  

    result = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=0)

    for obj in json_objects:
        parsed_obj = json.loads(obj)
        data_text = parsed_obj["data"]
        chunks = text_splitter.split_text(data_text)
        for chunk in chunks: 
        # Create the result dictionary with chunked data and the 'time' field
            result.append({"data": chunk, "time": parsed_obj["time"]})

    return result


def parse_pdf_to_txt(pdf_path, txt_path):
    """Parses a PDF file and stores its contents in a TXT file."""

    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    with open(txt_path, 'w') as txt_file:
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            txt_file.write(text)
            txt_file.write("\n\n")  # Add spacing between pages for readability

    pdf_file.close()


def add_to_knowledge_graph(G, embedding_model, threshold=0.75, batch_size=32):
    new_node_labels = []
    new_node_embeddings = []

    # Step 1: Split text into chunks and compute embeddings
    lines = count_lines_in_file("dynamic_kg_updation\out1.json")
    chunks=extract_data_and_time("dynamic_kg_updation\out1.json",lines-1)
    print(chunks)
    for i,chunk in enumerate(chunks):
        label = f"{chunk['time']} - Chunk {i+1}"
        if label not in G:  # Avoid adding duplicate nodes
            G.add_node(label, text=chunk['data'], embedding=embedding_model.encode([chunk['data']]))
            new_node_labels.append(label)
            new_node_embeddings.append(embedding_model.encode([chunk['data']]))

    # Step 2: Add edges for new nodes
    existing_node_embeddings = [G.nodes[node]["embedding"] for node in G.nodes]
    existing_node_labels = list(G.nodes)
    all_embeddings = np.vstack(existing_node_embeddings + new_node_embeddings)
    _add_edges_to_graph(G, existing_node_labels + new_node_labels, all_embeddings, threshold)
    import shutil

    # Move the processed file to the 'chunked_files' folder
    shutil.move(
        r"dynamic_kg_updation\files-for-indexing\\" + str(os.listdir('Harshvardhan/DocumentStore/files-for-indexing')[0]),
        r"dynamic_kg_updation\chunked_text"
    )

def _add_edges_to_graph(G, node_labels, node_embeddings, threshold):
    """
    Adds edges to the graph based on similarity threshold.

    Args:
        G (networkx.Graph): The knowledge graph.
        node_labels (list): List of node labels.
        node_embeddings (list): List of node embeddings.
        threshold (float): The similarity threshold for creating edges.

    Returns:
        None: Edges are added to the graph in place.
    """
    similarity_matrix = cosine_similarity(node_embeddings)
    num_nodes = len(node_labels)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                G.add_edge(node_labels[i], node_labels[j], weight=similarity)

    print(G.nodes)

def save_graph_to_json(G, fname):
    """
    Saves the knowledge graph to a JSON file, including node attributes and edge weights.
    """
    def serialize_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            return float(value)
        elif isinstance(value, (np.int32, np.int64)):
            return int(value)
        else:
            return value

    graph_data = {
        'nodes': [
            {
                'name': node,
                'attributes': {k: serialize_value(v) for k, v in G.nodes[node].items()}
            }
            for node in G.nodes
        ],
        'edges': [
            {
                'source': u,
                'target': v,
                'weight': float(G.edges[u, v].get('weight', 1.0))
            }
            for u, v in G.edges
        ]
    }

    with open(fname, 'w') as f:
        json.dump(graph_data, f, indent=2)


def main():
    while True:
        if len(os.listdir(r"dynamic_kg_updation\files-for-indexing")) == 0:
            continue 
        time.sleep(0.5)
        add_to_knowledge_graph(G, embedding_model)
        save_graph_to_json(G,r"dynamic_kg_updation\output.json")
        #The final Graph is troed into a new file
        #If a pre build graph was used then the original pre built graph itself is modified
if __name__=="__main__":
    main()
