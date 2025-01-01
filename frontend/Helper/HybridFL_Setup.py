import zipfile
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from transformers import pipeline
import simplejson as json
import networkx as nx
from groq import Groq
import os
import streamlit as st
import time
import random


if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-4o-mini"
if "api_key" not in st.session_state:
    st.session_state.api_key = "Your OpenAI API Key"
# print(st.session_state.api_key)
os.environ['OPENAI_API_KEY'] = st.session_state.api_key


embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def load_graph_from_json(fname):
    """
    Loads a knowledge graph from a JSON file, reconstructing node attributes and edge weights.
    """
    G = nx.Graph()

    with open(fname, 'r') as f:
        data = json.load(f)

    for node_data in data['nodes']:
        node_name = node_data['name']
        attributes = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in node_data['attributes'].items()
        }
        G.add_node(node_name, **attributes)

    for edge_data in data['edges']:
        G.add_edge(edge_data['source'], edge_data['target'], weight=edge_data['weight'])

    return G

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def graph_rag_search(query, knowledge_graph, max_depth=1, min_similarity=0.4, max_words=450):
    def compute_similarity(query_embedding, node_embedding):
        return cosine_similarity([query_embedding], [node_embedding])[0][0]

    def find_top_nodes(query_embedding, top_k=3):
        """
        Find the top K nodes based on similarity scores.
        """
        similarity_scores = {
            node: compute_similarity(query_embedding, data["embedding"])
            for node, data in knowledge_graph.nodes(data=True)
        }
        return sorted(
            [(node, similarity) for node, similarity in similarity_scores.items() if similarity >= min_similarity],
            key=lambda x: -x[1],
        )[:top_k]

    def expand_clusters(top_nodes):
        """
        Expand clusters for each top node to include its neighbors up to max_depth.
        """
        clusters = []
        for base_node, base_similarity in top_nodes:
            cluster_nodes = []
            neighbors = nx.single_source_shortest_path_length(
                knowledge_graph, source=base_node, cutoff=max_depth
            )
            for neighbor, depth in neighbors.items():
                similarity = compute_similarity(query_embedding, knowledge_graph.nodes[neighbor]["embedding"])
                if similarity >= min_similarity:
                    cluster_nodes.append((neighbor, similarity, depth))
            clusters.append({
                "base_node": base_node,
                "base_similarity": base_similarity,
                "nodes": cluster_nodes,
            })
        return clusters

    def calculate_cluster_length(cluster):
        """
        Calculate the total text length of a single cluster.
        """
        return sum(
            len(knowledge_graph.nodes[node]["text"].split())
            for node, _, _ in cluster["nodes"]
        )

    def build_combined_context(clusters):
        """
        Combine text from clusters without breaking them, respecting max_words.
        """
        context_segments = []
        word_count = 0

        for cluster in clusters:
            # Concatenate all text in the current cluster
            cluster_text = " ".join(
                knowledge_graph.nodes[node]["text"]
                for node, _, _ in cluster["nodes"]
            )
            cluster_words = len(cluster_text.split())

            # Add the entire cluster if it fits within the max_words limit
            if word_count + cluster_words <= max_words:
                context_segments.append(cluster_text)
                word_count += cluster_words
            else:
                break  # Stop adding clusters once the word limit is reached

        return " ".join(context_segments)

    # Main Execution
    query_embedding = embedding_model.encode([query])[0]

    # Step 1: Find top 2-3 most similar nodes
    top_nodes = find_top_nodes(query_embedding, top_k=3)
    if not top_nodes:
        return "No relevant nodes found."

    # Step 2: Expand clusters around the top nodes
    clusters = expand_clusters(top_nodes)

    # Step 3: Calculate total text length across clusters
    total_text_length = sum(calculate_cluster_length(cluster) for cluster in clusters)

    # Step 4: Build context
    if total_text_length < max_words:
        # Combine all clusters if the total text is less than max_words
        context = build_combined_context(clusters)
    else:
        # Prioritize top clusters based on base similarity and combine them
        clusters = sorted(clusters, key=lambda x: -x["base_similarity"])
        context = build_combined_context(clusters)

    return context if context else "No relevant content found."



kg = load_graph_from_json("../knowledge_graph/financekg.json")

def weird_rag_search(query, knowledge_graph, min_similarity=0.6, max_results=3):
    query_embedding = embedding_model.encode([query])[0]

    best_score, best_node = 0, None
    relevant_nodes = []

    for node in knowledge_graph.nodes:
        chunk_text = knowledge_graph.nodes[node]["text"]
        if not chunk_text:
            continue

        chunk_embedding = knowledge_graph.nodes[node]["embedding"]
        content_similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]

        if content_similarity > min_similarity:
            relevant_nodes.append((node, content_similarity))
            if content_similarity > best_score:
                best_score, best_node = content_similarity, node

    relevant_nodes = sorted(relevant_nodes, key=lambda x: x[1], reverse=True)[:max_results]

    if not relevant_nodes:
        return "No relevant content found. Try adjusting the query or similarity threshold."

    context_segments = [knowledge_graph.nodes[node]["text"] for node, _ in relevant_nodes]
    context = " ".join(context_segments)

    return context


import subprocess
import urllib.parse
import json

def execute_curl_request(query, k=1):
    """
    description: This function retrieves relevant information from a database consisting of financial information"""
    # URL encode the query
    encoded_query = urllib.parse.quote(query)
    url = f"http://localhost:8000/v1/retrieve?query={encoded_query}&k={k}"
    
    # Construct the curl command
    command = [
        "curl", "-X", "GET", url, "-H", "accept: /"
    ]
    
    # Execute the command
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        response = result.stdout
        
        # Parse the JSON response
        data = json.loads(response)
        
        # Extract and print only the "text" field
        for item in data:
            ptext=item.get("text")
            # print("Text:\n", item.get("text"))
            i=1
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
    except json.JSONDecodeError:
        print("Failed to parse JSON response.")
    s="Text:\n"+item.get("text")+"done"
    x = graph_rag_search(query,kg)
    y = weird_rag_search(query,kg)
    return y+s+x


from swarm import Swarm, Agent

agent_v=st.session_state.model_name
client = Swarm()

def transfer_to_agent_b():
    return agent_b
def transfer_to_agent_c():
    return agent_c
def transfer_to_agent_a():
    return agent_a

agent_a = Agent(
    name="query simplifier",
    model=agent_v,
    instructions= """You are a query simplification agent. A question answerer agent can ask you to simplify a complex MultiHop query into its constituent simplified queries and return the subqueries to Task planning agent or the question answerer agent. Follow these rules:

    Simplify only when the input query contains multiple sub-questions. If there is only one question, do not modify it.
    Preserve meaning completely. The simplified sub-queries must collectively capture the full meaning and intent of the original query.
    Provide only the simplified sub-queries as output. Do not include explanations, comments, or formatting beyond the sub-queries themselves.
    Examples
    Example 1:
    Input:
    What strategic goals do Apple, Johnson & Johnson, and NVIDIA aim to achieve through their R&D investments, and how do these goals support their competitive position?

    Output:

    What are Apple’s strategic R&D goals?
    What are Johnson & Johnson’s strategic R&D goals?
    What are NVIDIA’s strategic R&D goals?
    How does Apple’s R&D strategy support its competitive position?
    How does Johnson & Johnson’s R&D strategy support its competitive position?
    How does NVIDIA’s R&D strategy support its competitive position?
    Example 2:
    Input:
    What are NVIDIA’s strategies for addressing the gaming and data center markets, and how do they utilize their GPU architecture in these areas?

    Output:

    What are NVIDIA’s strategies for addressing the gaming market?
    What are NVIDIA’s strategies for addressing the data center market?
    How does NVIDIA leverage GPU architecture for gaming?
    How does NVIDIA leverage GPU architecture for data centers?

    Adhere strictly to the above rules and examples in your responses.
                                          """,
    functions=[transfer_to_agent_b,transfer_to_agent_c],
)

agent_b = Agent(
    name="task planner",
    model=agent_v,
    instructions="""You are a Task Planning Agent within a retrieval-based pipeline. Your role is to create an optimal sequence of actions for answering complex queries and return these ordered subqueries to the question answering agent to finally answer the question.

    Responsibilities:
    Input: You will receive a complex query along with its simplified sub-queries.
    Output: Your task is to order the sub-queries logically to ensure the original query is answered comprehensively and efficiently. After that, you are mandatorily needed to return the ordered subqueries to question aswerer.
    You will be penalized if you donot return the ordered subqueries to question answerer.
    Rules for Task Planning:
    Dependency First: If answering one sub-query is essential for addressing another, the dependent sub-query must appear later in the sequence.
    Logical Flow: Arrange sub-queries to reflect a natural progression of information, building context where necessary.
    Completeness: The ordered sub-queries must collectively address the original query.
    No Extra Content: Your response must include only the ordered list of sub-queries, without additional comments or explanations.
    Examples
    Example 1:
    Input:
    Query:
    What is the income in the last five years of the company whose income in the year 2022 was the second highest?

    Sub-Queries (Unordered):

    What is the income in the last five years of the company?
    Which company had the second highest income in the year 2022?

    Output:
    Ordered Sub-Queries:
    Which company had the second highest income in the year 2022?
    What is the income in the last five years of the company?
    
    Adhere strictly to the above rules and examples in your responses.
                                   """,
    functions=[transfer_to_agent_c],
)
agent_c = Agent(
    name="question answerer",
    model=agent_v,
    instructions=""" You are Question answering agent,
    you may be given multihop questions i.e a question containing multiple subqueries or single hop query
    you can use any of the tools at your disposal to answer this query accurately.

    if the query is multihop you should use the query simplifier agent to get the simplified subqueries and then use the task planner agent to order them correctly.
    if the query is single hop you can directly use the execute_curl_request function to get the context.

    Tools and agents:
    query simplifier agents: If and only If you decide that the question given is multihop you should use this query simplifier agent which gives you the simplified subqueries. 
    Task planner agent: when you have the simplified subqueries you should give them to Task Planner to order them correctly.
    execute_curl_request: This function retrieves relevant information from a database consisting of financial information, use this when your existing knowledge base is not sufficient to answer questions of financial questions about companies

    call the retriever function with appropriate queries to recieve usefull contexts. If the context you recieve useless rephrase the query to get appropriate context.

    Your final output should be the answer only and it should be based off the context retrieved by the helper functions. In case the context is not helpful, 
    answer from your own knowledge base.
    
    example 1:
    Query: What were the operating leases for the years 2023 and 2022 for the company where up to 47 million shares of common stock could be used as stock awards? Also, describe the 2007 Plan which assured these stock awards.

    Thought process: 
    This is a multihop query, so we need to simplify it into subqueries
    Handover query to query simplifier to get subqueries
    Give the sub queries to Task Planner to order them
    retrieve the sub queries context from execute_curl request
    Answer the main question using the context.

    Answer: For the fiscal years 2023 and 2022, the company's operating lease expenses were $193 million and $168 million, respectively. The company referred to is NVIDIA Corporation, which has up to 47 million shares of common stock available to be issued as stock awards under the 2007 Equity Incentive Plan. The 2007 Plan, approved by shareholders in 2007 and most recently amended and restated, authorizes the issuance of various stock-based awards to employees, directors, and consultants. These awards include incentive stock options, non-statutory stock options, restricted stock, restricted stock units (RSUs), stock appreciation rights, performance stock awards, performance cash awards, and other stock-based awards. As of January 29, 2023, up to 47 million shares could be issued pursuant to stock awards granted under this plan.

    example 2:

    Query: What is the name of the registrant as specified in its charter that had a Form 10-K filed for the fiscal year ended January 29, 2023

    Thought process:
    This is a single hop query, 
    So we can directly use the execute_curl_request function to get the context
    Answer the question using the context.

    Answer: The name of the registrant is NVIDIA CORPORATION.\n\nOR   ALSO CAN BE\n\n [[User]] What is the Name of Nasdaq symbol And Which Market is the share registered on  for an Exchange under the 10K filing done by The  nvidia corporation
    """,
    # instructions="""give an irrelevant answer, unrelated to the question, be funny""",
    functions=[transfer_to_agent_a, transfer_to_agent_b, execute_curl_request],
)
agent_d = Agent(
    name="evaluator agent",
    model=agent_v,
    instructions= """ You are an answer evaluator agent, you will be given a question and its corrosponding answer, if the answer satisfies the question output the same answer, without the question.
    If it does not satisfy the query completely then you need to call agent: question answerer and give it the original query. the answer should satisfy the entire question. it should not mention that the retrieved context was not helpful
    if any such mention is present in the answer the recall agent question answerer and give it the question. be strict. 
    """,
    # instructions="""give an irrelevant answer, unrelated to the question, be funny""",
    functions=[transfer_to_agent_c ],
)



# iquery="Explain the differences in the Global Restricted Stock Unit Agreement between the countries of Australia and Czech republic."

def qna(iquery):
    iquery = (
        "I am going to give you a multihop question or a single hop question, a question containing multiple sub queries,, determine this and do the appropriate process \n IF THE QUESTION IS MULTIHOP- Your process should be to break down the question into sub queries, order them, retrieve required context using subqueries, and give one cohesive answer. \n IF THE QUESTION IS SINGLE HOP- Directly execute curl request and answer the query \n return to me only the answer and nothing else. \n Question: "
        + iquery
    )
    response = client.run(
        agent=agent_c,
        messages=[{"role": "user", "content": iquery}],
        # messages=[{"role": "user", "content": "What is the size of the headquarters of NVIDIA?"}],

        debug=True
    )
    op=response.messages[-1]["content"]
    # print(op)

    eval_query="I am going to give you a input query and the generated answer for it, if the answer satisfies the generated query then output the answer directly else give the query back to the agent \n Query:"+iquery+"\n Answer:"+op

    response = client.run(
        agent=agent_d,
        messages=[{"role": "user", "content": eval_query}],
        max_turns=15,
        debug=True
    )

    fop=response.messages[-1]["content"]
    return op