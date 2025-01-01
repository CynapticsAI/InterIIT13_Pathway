# Imports 
from __future__ import annotations

import os
import json
import openai
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import PathwayVectorClient
import os
import streamlit as st
import streamlit_scrollable_textbox as stx

import subprocess
import urllib.parse

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from langchain_core.callbacks import (
    CallbackManagerForChainRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from pydantic import Field

from langchain.chains.base import Chain
from langchain.chains.flare.prompts import (
    PROMPT,
    QUESTION_GENERATOR_PROMPT,
    FinishedOutputParser,
)
from langchain.chains.llm import LLMChain
from langchain_core.vectorstores import VectorStore
import os 

from langchain_community.tools import DuckDuckGoSearchRun
import json
import PyPDF2
from streamlit_modal import Modal
import time
import random
subquery_list = []
current_response_list = []
tool_calls_dict =[]

if "model_choice" not in st.session_state:
    st.session_state.model_choice = "gpt-4o-mini"
if "api_key" not in st.session_state:
    st.session_state.api_key = "Your OpenAI API Key"



# LLM

os.environ["OPENAI_API_KEY"] = st.session_state.api_key

llm = ChatOpenAI(model="Your OpenAI API Key", logprobs=True)


# Pathway Client

client = PathwayVectorClient(url="http://localhost:8000")

# FLARE Setup

def _extract_tokens_and_log_probs(response: AIMessage) -> Tuple[List[str], List[float]]:
    """Extract tokens and log probabilities from chat model response."""
    tokens = []
    log_probs = []
    for token in response.response_metadata["logprobs"]["content"]:
        tokens.append(token["token"])
        log_probs.append(token["logprob"])

    return tokens, log_probs


class QuestionGeneratorChain(LLMChain):
    """Chain that generates questions from uncertain spans."""

    prompt: BasePromptTemplate = QUESTION_GENERATOR_PROMPT
    """Prompt template for the chain."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def input_keys(self) -> List[str]:
        """Input keys for the chain."""
        return ["user_input", "context", "response"]


def _low_confidence_spans(
    tokens: Sequence[str],
    log_probs: Sequence[float],
    min_prob: float,
    min_token_gap: int,
    num_pad_tokens: int,
) -> List[str]:
    _low_idx = np.where(np.exp(log_probs) < min_prob)[0]
    low_idx = [i for i in _low_idx if re.search(r"\w", tokens[i])]
    if len(low_idx) == 0:
        return []
    spans = [[low_idx[0], low_idx[0] + num_pad_tokens + 1]]
    for i, idx in enumerate(low_idx[1:]):
        end = idx + num_pad_tokens + 1
        if idx - low_idx[i] < min_token_gap:
            spans[-1][1] = end
        else:
            spans.append([idx, end])
    return ["".join(tokens[start:end]) for start, end in spans]


class FlareChain(Chain):
    """Chain that combines a retriever, a question generator,
    and a response generator.

    See [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983) paper.
    """

    question_generator_chain: Runnable
    """Chain that generates questions from uncertain spans."""
    response_chain: Runnable
    """Chain that generates responses from user input and context."""
    output_parser: FinishedOutputParser = Field(default_factory=FinishedOutputParser)
    """Parser that determines whether the chain is finished."""
    retriever: VectorStore
    """Retriever that retrieves relevant documents from a user input."""
    min_prob: float = 0.2
    """Minimum probability for a token to be considered low confidence."""
    min_token_gap: int = 5
    """Minimum number of tokens between two low confidence spans."""
    num_pad_tokens: int = 2
    """Number of tokens to pad around a low confidence span."""
    max_iter: int = 10
    """Maximum number of iterations."""
    start_with_retrieval: bool = True
    """Whether to start with retrieval."""

    @property
    def input_keys(self) -> List[str]:
        """Input keys for the chain."""
        return ["user_input"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys for the chain."""
        return ["response"]

    def _do_generation(
        self,
        questions: List[str],
        user_input: str,
        response: str,
        _run_manager: CallbackManagerForChainRun,
    ) -> Tuple[str, bool]:
        callbacks = _run_manager.get_child()
        docs = []
        query_context_dict = {}
        if response.lower() == "":
                current_response_list.append("0")
        else:
            current_response_list.append(response)

        for question in questions:
            d = self.retriever.similarity_search(question,k=5)
            docs.extend(d)
            query_context_dict[question] = d 

        subquery_list.append(query_context_dict)
        context = "\n\n".join(d.page_content for d in docs)
        result = self.response_chain.invoke(
            {
                "user_input": user_input,
                "context": context,
                "response": response,
            },
            {"callbacks": callbacks},
        )
        if isinstance(result, AIMessage):
            result = result.content
        marginal, finished = self.output_parser.parse(result)
        return marginal, finished

    def _do_retrieval(
        self,
        low_confidence_spans: List[str],
        _run_manager: CallbackManagerForChainRun,
        user_input: str,
        response: str,
        initial_response: str,
    ) -> Tuple[str, bool]:
        question_gen_inputs = [
            {
                "user_input": user_input,
                "current_response": initial_response,
                "uncertain_span": span,
            }
            for span in low_confidence_spans
        ]
        callbacks = _run_manager.get_child()
        if isinstance(self.question_generator_chain, LLMChain):
            question_gen_outputs = self.question_generator_chain.apply(
                question_gen_inputs, callbacks=callbacks
            )
            questions = [
                output[self.question_generator_chain.output_keys[0]]
                for output in question_gen_outputs
            ]
        else:
            questions = self.question_generator_chain.batch(
                question_gen_inputs, config={"callbacks": callbacks}
            )
        _run_manager.on_text(
            f"Generated Questions: {questions}", color="yellow", end="\n"
        )
        return self._do_generation(questions, user_input, response, _run_manager)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        user_input = inputs[self.input_keys[0]]

        response = ""

        for i in range(self.max_iter):
            _run_manager.on_text(
                f"Current Response: {response}", color="blue", end="\n"
            )
            _input = {"user_input": user_input, "context": "", "response": response}
            tokens, log_probs = _extract_tokens_and_log_probs(
                self.response_chain.invoke(
                    _input, {"callbacks": _run_manager.get_child()}
                )
            )
            low_confidence_spans = _low_confidence_spans(
                tokens,
                log_probs,
                self.min_prob,
                self.min_token_gap,
                self.num_pad_tokens,
            )
            initial_response = response.strip() + " " + "".join(tokens)
            if not low_confidence_spans:
                response = initial_response
                final_response, finished = self.output_parser.parse(response)
                if finished:
                    return {self.output_keys[0]: final_response}
                continue

            marginal, finished = self._do_retrieval(
                low_confidence_spans,
                _run_manager,
                user_input,
                response,
                initial_response,
            )
            response = response.strip() + " " + marginal
            if finished:
                break
        # current_response = response
        return {self.output_keys[0]: response}

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, max_generation_len: int = 32, **kwargs: Any
    ) -> FlareChain:
        """Creates a FlareChain from a language model.

        Args:
            llm: Language model to use.
            max_generation_len: Maximum length of the generated response.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            FlareChain class with the given language model.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI is required for FlareChain. "
                "Please install langchain-openai."
                "pip install langchain-openai"
            )
        llm = ChatOpenAI(model=st.session_state.model_choice,max_tokens=max_generation_len, logprobs=True, temperature=0)
        response_chain = PROMPT | llm
        question_gen_chain = QUESTION_GENERATOR_PROMPT | llm | StrOutputParser()
        return cls(
            question_generator_chain=question_gen_chain,
            response_chain=response_chain,
            **kwargs,
        )

# Flare Chain

flare = FlareChain.from_llm(
    llm=llm,
    retriever=client,
    max_generation_len=300,
    min_prob=0.45,
    max_iter=10,
    verbose=True,
)

def generate_flare_output(input_text):
    output = flare.run(input_text)
    return output

# Tools

tools = [
    {
        "type": "function",
        "function": {
            "name": "math_operations",
            "description": "Performs basic arithmetic operations: addition, subtraction, multiplication, and division. Use this for mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number."
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number."
                    },
                    "operation_code": {
                        "type": "integer",
                        "description": "The operation to perform (1: Add, 2: Subtract, 3: Multiply, 4: Divide)."
                    }
                },
                "required": ["a", "b", "operation_code"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "parse_pdf_to_json",
            "description": "Parses the contents of a PDF file and saves it in a structured JSON format. Use this for converting PDFs to machine-readable data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "The file path of the PDF to parse."
                    },
                    "json_path": {
                        "type": "string",
                        "description": "The file path to save the output JSON."
                    }
                },
                "required": ["pdf_path", "json_path"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_duckduckgo",
            "description": "Performs a web search using DuckDuckGo. Use this to retrieve search results for a specific keyword or phrase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The search term or phrase."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum number of search results to retrieve."
                    }
                },
                "required": ["keyword"],
                "additionalProperties": False
            }
        }
    }
]

# Helper Functions

###
def create_messages(query):
  return [
        {
            "role": "system",
            "content": "You are a skilled and empathetic assistant, adept at solving user problems efficiently and accurately. Use your deep understanding of context, reasoning, and tools at your disposal to provide solutions tailored to the user's needs. When appropriate, ask clarifying questions to ensure a thorough response. Respond concisely and clearly, maintaining a friendly and professional tone. Always prioritize the user's objectives while explaining your reasoning where necessary."
        },
        {
            "role": "user",
            "content": query
        }
    ]

def classify_query(query):
    prompt = f"""
    The following query needs to be classified into one of two categories: 'Finance' or 'Legal'.

    Query: "{query}"

    Categories:
    1. Finance
    2. Legal

    Please classify the query into one of the categories. Just give answers in one word.
    """

    msg = create_messages(prompt)


    try:
        response = openai.chat.completions.create(
            model=st.session_state.model_choice,
            messages=msg,
        )

        classification = response.choices[0].message.content

        return classification
    except Exception as e:
        return f"Error: {str(e)}"

###
def is_satisfied(query,response):
    evaluation_prompt = f"""
    Evaluate the following:

    User Query: '{query}'
    LLM Response: '{response}'

    Does the LLM response fully and accurately satisfy the user's query based on the provided context?

    Respond with "Yes" if the response completely satisfies the user's query. Respond with "No" if it does not.
    """
    msg = create_messages(evaluation_prompt)

    try:
        response = openai.chat.completions.create(
            model=st.session_state.model_choice,
            messages=msg,
        )

        classification = response.choices[0].message.content

        return classification
    except Exception as e:
        return f"Error: {str(e)}"

###
def updated_reponse_generator(query,extra_context,prev_response):
    new_prompt = f"""
    Using the provided resource below, improve or generate a complete response to the user query:

    Resource: {extra_context}

    User Query: {query}

    Previous Response: {prev_response}

    Consider the following:
    1. Ensure the resource is fully utilized to address the query.
    2. If the previous response is incomplete or incorrect, provide a revised and complete response.
    3. Maintain a concise, user-friendly, and accurate tone in your response.
    """

    msg = create_messages(new_prompt)

    try:
        response = openai.chat.completions.create(
            model=st.session_state.model_choice,
            messages=msg,
        )

        classification = response.choices[0].message.content

        return classification
    except Exception as e:
        return f"Error: {str(e)}"

###
def tool_calls(response):
    choices = response.choices  
    parsed_data = []

    for choice in choices:
        for tool_call in choice.message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            tool_name = tool_call.function.name
            parsed_data.append({"tool_name": tool_name, "arguments": arguments})

    return parsed_data

# Tool Functions

###

def search_duckduckgo(keyword, max_results=5):
    "Performs a web search using DuckDuckGo. Use this to retrieve search results for a specific keyword or phrase."
    print(f"Tool called with input: {keyword}")
    search = DuckDuckGoSearchRun()
    # results = DDGS().text(keyword, max_results=max_results)
    results = search.invoke(keyword, max_results=max_results)
    return results

###


def parse_pdf_to_json(pdf_path, json_path):
    """Parses a PDF file and stores its contents in a JSON file."""

    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    data = []

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        data.append({"page": page_num + 1, "content": text})

    pdf_file.close()

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


###

def math_operations(a, b, operation_code):
    if operation_code == 1:
        return a + b
    elif operation_code == 2:
        return a - b
    elif operation_code == 3:
        return a * b
    elif operation_code == 4:
        return a / b if b != 0 else "Error: Division by zero"
    else:
        return "Error: Invalid operation code"
    
# Flare Agent Function

def Agent(query):

  #FLARE output
  response = generate_flare_output(query)

  #check if LLM is satisfied with answer
  status = is_satisfied(query,response)
  if status.lower() == 'yes':
    return response , []

  elif status.lower() == 'no':
    test_prompt = f"""
    The LLM response to the user query "{query}" was "{response}".
    Ensure the continuation of the response uses relevant tool calls to provide accurate and actionable outputs.
    """
    msg = create_messages(test_prompt)
    further_response = openai.chat.completions.create(
            model=st.session_state.model_choice,
            messages=msg,
            tools=tools,
          )

    parsed_data = tool_calls(further_response)
    tool_calls_dict = parsed_data

    for data in parsed_data:
      if data['tool_name'].lower() == 'search_duckduckgo':
        extra_context = search_duckduckgo(data['arguments']['keyword'])
      elif data['tool_name'].lower() == 'math_operations':
        extra_context = parse_pdf_to_json(data['arguments']['pdf_path'],data['arguments']['json_path'])
      elif data['tool_name'].lower() == 'math_operations':
        extra_context = math_operations(data['arguments']['a'],data['arguments']['b'],data['arguments']['operation_code'])


    final_response = updated_reponse_generator(query,extra_context,response)

    return final_response , tool_calls_dict
  

def regenerated_answer(prev_response,regenerated_query):
    context = client.similarity_search(regenerated_query,k=5)
    prompt = f"""
    *Previous Response:*
    {prev_response}

    *Regenerated Query:*
    {regenerated_query}

    *Extra Context:*
    {context}

    ### *Instructions for Updating the Previous Response:*

    Using the provided previous response, regenerated query, and extra context, please update and/or correct the previous response as necessary. Consider the following when updating:

    1. *Addressing any inconsistencies:* If the previous response is missing details or misinterprets the query, provide a more accurate or thorough answer.
    2. *Improving clarity:* If the original response was unclear or confusing, rewrite it to make it more understandable.
    3. *Including the extra context:* Make sure to integrate any relevant additional context provided to ensure the response is more tailored to the user's needs.
    4. *Ensuring relevance:* If the regenerated query or extra context introduces new factors or nuances, update the response accordingly to ensure it’s fully aligned with the updated query.

    Provide the revised version of the answer based on the extra context.
    """
    msg = create_messages(prompt)
    try:
        response = openai.chat.completions.create(
            model=st.session_state.model_choice,
            messages=msg,
        )

        classification = response.choices[0].message.content

        return classification
    except Exception as e:
        return f"Error: {str(e)}"